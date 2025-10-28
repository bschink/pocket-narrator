import os, math, time, argparse, json, random
import torch
import torch.nn as nn
from torch.optim import AdamW

from pocket_narrator.model import PocketNarratorModel, ModelConfig
from pocket_narrator.data_loader import build_dataloaders
from pocket_narrator.evaluate import perplexity

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def cosine_with_warmup(step, max_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="models/ts_run1")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--sample_prompt", type=str, default="Tom and Jane are friends.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Data
    train_loader, val_loader, vocab_size, eos_id, decode_fn = build_dataloaders(
        data_dir=args.data_dir,
        split_ratio=0.98,
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # 2) Model
    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        pdrop=0.0,
        tie_weights=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PocketNarratorModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # 3) Optim
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    # 4) Train
    step, best_ppl = 0, float("inf")
    t0 = time.time()
    model.train()
    while step < args.max_steps:
        for xb, yb in train_loader:
            if step >= args.max_steps: break
            lr_now = cosine_with_warmup(step, args.max_steps, args.lr, args.warmup_steps)
            for g in optim.param_groups: g['lr'] = lr_now
            xb, yb = xb.to(device), yb.to(device)

            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)

            if (step+1) % 50 == 0:
                dt = time.time() - t0
                print(f"[step {step+1}] loss={loss.item():.3f} lr={lr_now:.2e} dt={dt:.1f}s")
                t0 = time.time()

            if (step+1) % args.eval_every == 0:
                val_ppl = perplexity(model, val_loader, device=device)
                print(f"[eval] val perplexity={val_ppl:.2f}")
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
                               os.path.join(args.out_dir, "model_best.pt"))
                    print(f"[ckpt] best improved → saved model_best.pt (ppl={best_ppl:.2f})")
            step += 1

    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
               os.path.join(args.out_dir, "model_last.pt"))
    print("[done] saved model_last.pt")

    # 5) Acceptance: prompt → generate a sample
    try:
        # Build a toy input from the prompt using a trivial whitespace mapping coming from data_loader
        # (Your build_dataloaders returns a decode_fn; for encode, use the data_loader's helper if available.
        # If not, a minimal fallback is to split and map unknown to 0.)
        from pocket_narrator.data_loader import encode_prompt  # optional helper if Yumna implements it
        if 'encode_prompt' in globals():
            ids = encode_prompt(args.sample_prompt)
        else:
            ids = [1]  # start with something valid; data_loader should provide a real encoder soon
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=60, temperature=0.9, top_k=40, eos_token_id=eos_id)
        print("\n=== SAMPLE ===")
        print(decode_fn(y[0].tolist()))
        print("==============\n")
    except Exception as e:
        print(f"[warn] sample generation skipped: {e}")

if __name__ == "__main__":
    main()
