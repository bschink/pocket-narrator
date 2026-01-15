# the main mamba_evaluation without bleu, rouge, recall and F1
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import wandb
import re
from transformers import PreTrainedTokenizerFast

from pocket_narrator.models.mamba.config_utils import load_yaml
from pocket_narrator.models.mamba.mamba_model import MambaConfig, MambaLM
from pocket_narrator.models.mamba.mamba_trainer import create_dataloaders

"""
How to run (from the PROJECT ROOT): 
1. run evaluation

python -m pocket_narrator.models.mamba.mamba_evaluation.py \
  --checkpoint results/mamba_tinystories_1M/mamba_best.pt \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_dir tokenizers/tinystories_1M \
  --dataset roneneldan/TinyStories-1M \
  --split validation \
  --max_examples 256

  
  python -m pocket_narrator.models.mamba.mamba_evaluation \
  --model_config configs/mamba_tinystories_2k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_2k/tokenizer.yaml \
  --checkpoint results/mamba_tinystories_2k/mamba_best.pt \
  --num_workers 0

python -m pocket_narrator.models.mamba.mamba_evaluation \
  --model_config configs/mamba_tinystories_4k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_4k/tokenizer.yaml \
  --checkpoint results/mamba_tinystories_4k/mamba_best.pt \
  --num_workers 0

python -m pocket_narrator.models.mamba.mamba_evaluation \
  --model_config configs/mamba_tinystories_8k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_8k/tokenizer.yaml \
  --checkpoint results/mamba_tinystories_8k/mamba_best.pt \
  --num_workers 0  

python -m pocket_narrator.models.mamba.mamba_evaluation \
  --model_config configs/mamba_tinystories_10k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_10k/tokenizer.yaml \
  --checkpoint results/mamba_tinystories_10k/mamba_best.pt \
  --num_workers 0  

python -m pocket_narrator.models.mamba.mamba_evaluation \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_config configs/mamba_tinystories_1M/tokenizer.yaml \
  --checkpoint results/mamba_tinystories_1M/mamba_best.pt \
  --num_workers 0    
"""

# Text-metrics helpers
def _word_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def distinct_n(text: str, n: int = 2) -> float:
    tokens = _word_tokenize(text)
    ng = _ngrams(tokens, n)
    if not ng:
        return 0.0
    return len(set(ng)) / len(ng)

def repetition_rate(text: str, n: int = 2) -> float:
    return 1.0 - distinct_n(text, n=n)


@torch.no_grad()
def generate_samples(
    model: MambaLM,
    tokenizer: PreTrainedTokenizerFast,
    mcfg: MambaConfig,
    device: torch.device,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
):
    model.eval()
    samples = []

    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        eos_token_id = tokenizer.pad_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.size(1) > mcfg.seq_len:
            input_ids = input_ids[:, -mcfg.seq_len :]

        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

        tokens = _word_tokenize(text)
        len_tokens = len(tokens)

        d2 = distinct_n(text, n=2)
        d3 = distinct_n(text, n=3)
        r2 = repetition_rate(text, n=2)
        r3 = repetition_rate(text, n=3)

        samples.append(
            {
                "prompt": prompt,
                "generation": text,
                "len_tokens": len_tokens,
                "distinct_2": d2,
                "distinct_3": d3,
                "rep_2": r2,
                "rep_3": r3,
            }
        )
    return samples


# -------------------------
# LM evaluation
# -------------------------
@torch.no_grad()
def evaluate_lm_loss_ppl(
    model: MambaLM,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = out["loss"]  # mean CE over non-ignored positions

        # For a more token-aware aggregate, count non-pad tokens
        pad_id = model.config.pad_token_id
        n_tokens = (labels != pad_id).sum().item()

        # loss is mean per token (over non-ignored), so multiply by tokens
        total_loss += loss.item() * max(n_tokens, 1)
        total_tokens += max(n_tokens, 1)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")  # avoid overflow
    return {"eval_loss": avg_loss, "eval_ppl": ppl, "eval_tokens": float(total_tokens)}


def _build_mcfg_from_sources(
    model_cfg: Dict[str, Any],
    ckpt_payload: Optional[Dict[str, Any]],
) -> MambaConfig:
    """
    Priority:
    1) checkpoint["config"] (if present)
    2) model.yaml
    """
    if ckpt_payload is not None and "config" in ckpt_payload and isinstance(ckpt_payload["config"], dict):
        c = ckpt_payload["config"]
        return MambaConfig(
            vocab_size=int(c["vocab_size"]),
            d_model=int(c["d_model"]),
            n_layers=int(c["n_layers"]),
            seq_len=int(c.get("seq_len", 256)),
            d_state=int(c.get("d_state", 16)),
            d_conv=int(c.get("d_conv", 3)),
            expand=int(c.get("expand", 2)),
            pad_token_id=int(c.get("pad_token_id", 0)),
            dropout=float(c.get("dropout", 0.0)),
            layer_norm_eps=float(c.get("layer_norm_eps", 1e-5)),
        )

    return MambaConfig(
        vocab_size=int(model_cfg["vocab_size"]),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        seq_len=int(model_cfg.get("seq_len", 256)),
        d_state=int(model_cfg.get("d_state", 16)),
        d_conv=int(model_cfg.get("d_conv", 3)),
        expand=int(model_cfg.get("expand", 2)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        layer_norm_eps=float(model_cfg.get("layer_norm_eps", 1e-5)),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MambaLM checkpoint.")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model.yaml")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="Path to tokenizer.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt (mamba_best.pt, etc.)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch_size (else from tokenizer/training setup)")
    parser.add_argument("--num_workers", type=int, default=2)

    # generation controls
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)

    # prompts / outputs
    parser.add_argument("--prompts_file", type=str, default=None, help="Optional txt file with one prompt per line")
    parser.add_argument("--output_dir", type=str, default="./results/mamba_eval")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # wandb config
    parser.add_argument("--wandb_project", type=str, default="Mamba")
    parser.add_argument("--wandb_entity", type=str, default="once-upon-a-prompt")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Args:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_cfg = load_yaml(args.model_config)
    tok_cfg = load_yaml(args.tokenizer_config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_payload = torch.load(str(ckpt_path), map_location="cpu")

    mcfg = _build_mcfg_from_sources(model_cfg, ckpt_payload)
    model = MambaLM(mcfg).to(device)
    model.load_state_dict(ckpt_payload["model_state_dict"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] Model parameters: {n_params:,}")

    # Data
    lm_dataset_dir = tok_cfg["lm_dataset_dir"]
    bs = args.batch_size if args.batch_size is not None else 1
    train_dl, eval_dl = create_dataloaders(
    lm_dataset_dir=lm_dataset_dir,
    train_batch_size=bs,
    eval_batch_size=bs,
    num_workers=args.num_workers,
)

    # Tokenizer
    tokenizer_dir = tok_cfg.get("save_dir") or tok_cfg.get("tokenizer_dir")
    if tokenizer_dir is None:
        raise ValueError("tokenizer_config must contain save_dir or tokenizer_dir.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    # Prompts
    if args.prompts_file:
        p = Path(args.prompts_file)
        prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        prompts = [
            "Once upon a time there was a little dragon",
            "In a small village lived a kind robot",
            "The princess wanted to learn how to code",
        ]

    # W&B init
    run = None
    run_name = args.wandb_run_name or f"eval_{model_cfg.get('name','mamba')}_{ckpt_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not args.no_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "mode": "evaluation",
                "checkpoint": str(ckpt_path),
                "lm_dataset_dir": lm_dataset_dir,
                "batch_size": bs,
                "vocab_size": mcfg.vocab_size,
                "d_model": mcfg.d_model,
                "n_layers": mcfg.n_layers,
                "seq_len": mcfg.seq_len,
                "d_state": mcfg.d_state,
                "d_conv": mcfg.d_conv,
                "expand": mcfg.expand,
                "dropout": mcfg.dropout,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "num_params": n_params,
            },
        )

    # Evaluate LM metrics
    lm_metrics = evaluate_lm_loss_ppl(model, eval_dl, device)
    print(f"[RESULT] eval_loss={lm_metrics['eval_loss']:.4f} eval_ppl={lm_metrics['eval_ppl']:.2f} tokens={int(lm_metrics['eval_tokens'])}")

    # Generate samples + text metrics
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        mcfg=mcfg,
        device=device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Aggregate sample metrics
    if len(samples) > 0:
        mean_len = sum(s["len_tokens"] for s in samples) / len(samples)
        mean_d2 = sum(s["distinct_2"] for s in samples) / len(samples)
        mean_d3 = sum(s["distinct_3"] for s in samples) / len(samples)
        mean_r2 = sum(s["rep_2"] for s in samples) / len(samples)
        mean_r3 = sum(s["rep_3"] for s in samples) / len(samples)
    else:
        mean_len = mean_d2 = mean_d3 = mean_r2 = mean_r3 = 0.0

    # Save local outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "checkpoint": str(ckpt_path),
        "eval_loss": lm_metrics["eval_loss"],
        "eval_ppl": lm_metrics["eval_ppl"],
        "eval_tokens": lm_metrics["eval_tokens"],
        "mean_len_tokens": mean_len,
        "mean_distinct_2": mean_d2,
        "mean_distinct_3": mean_d3,
        "mean_rep_2": mean_r2,
        "mean_rep_3": mean_r3,
        "samples": samples,
        "config": mcfg.__dict__,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote: {out_dir / 'evaluation.json'}")

    # W&B logging
    if run is not None:
        wandb.log(
            {
                "eval_loss": lm_metrics["eval_loss"],
                "eval_ppl": lm_metrics["eval_ppl"],
                "eval_tokens": lm_metrics["eval_tokens"],
                "mean_len_tokens": mean_len,
                "mean_distinct_2": mean_d2,
                "mean_distinct_3": mean_d3,
                "mean_rep_2": mean_r2,
                "mean_rep_3": mean_r3,
            }
        )

        table = wandb.Table(
            columns=["prompt", "generation", "len_tokens", "distinct_2", "distinct_3", "rep_2", "rep_3"]
        )
        for s in samples:
            table.add_data(
                s["prompt"],
                s["generation"],
                s["len_tokens"],
                s["distinct_2"],
                s["distinct_3"],
                s["rep_2"],
                s["rep_3"],
            )
        wandb.log({"eval_samples": table})

        wandb.finish()


if __name__ == "__main__":
    main()
