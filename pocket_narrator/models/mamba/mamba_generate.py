# pocket_narrator/models/mamba/mamba_generate.py

import argparse
import os
from datetime import datetime
from typing import Optional, List

import torch
from transformers import PreTrainedTokenizerFast

import wandb
from transformers import PreTrainedTokenizerFast

from pocket_narrator.models.mamba.config_utils import load_yaml
from pocket_narrator.models.mamba.mamba_model import MambaLM, MambaConfig

"""cd /Users/kosaralehosseini/Documents/Project/pocket-narrator

problem : nl -ba configs/mamba_tinystories_10k/tokenizer.yaml | sed -n '1,40p'


python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_2k/mamba_best.pt \
  --model_config configs/mamba_tinystories_2k/model.yaml \
  --tokenizer_dir tokenizers/tinystories_2k \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 128

python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_4k/mamba_best.pt \
  --model_config configs/mamba_tinystories_4k/model.yaml \
  --tokenizer_dir tokenizers/tinystories_4k \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 128

python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_8k/mamba_best.pt \
  --model_config configs/mamba_tinystories_8k/model.yaml \
  --tokenizer_dir tokenizers/tinystories_8k \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 128  
  

python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_10k/mamba_best.pt \
  --model_config configs/mamba_tinystories_10k/model.yaml \
  --tokenizer_dir tokenizers/tinystories_10k \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 128  
  

python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_1M/mamba_best.pt \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_dir tokenizers/tinystories_1M \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 128  
  """



def load_prompts(prompt: Optional[str], prompts_file: Optional[str]) -> List[str]:
    if prompt is not None:
        return [prompt]

    if prompts_file is None:
        raise ValueError("Either --prompt or --prompts_file must be provided.")

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError("prompts_file is empty.")

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Mamba TinyStories model."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Mamba")
    parser.add_argument("--wandb_entity", type=str, default="once-upon-a-prompt")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    
    # Setup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model_cfg = load_yaml(args.model_config)
    mcfg = MambaConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        seq_len=model_cfg.get("seq_len", 256),
        d_state=model_cfg.get("d_state", 16),
        d_conv=model_cfg.get("d_conv", 3),
        expand=model_cfg.get("expand", 2),
        pad_token_id=model_cfg.get("pad_token_id", 0),
        dropout=model_cfg.get("dropout", 0.0),
    )

    model = MambaLM(mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt_cfg = ckpt.get("config", {})
    if "vocab_size" in ckpt_cfg:
        mcfg.vocab_size = int(ckpt_cfg["vocab_size"])
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        eos_token_id = tokenizer.pad_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    prompts = load_prompts(args.prompt, args.prompts_file)

    
    # W&B init
   
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode="disabled" if args.no_wandb else None,
        config={
            "checkpoint": args.checkpoint,
            "model_config": args.model_config,
            "tokenizer_dir": args.tokenizer_dir,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_prompts": len(prompts),
        },
    )

    
    # Generation
    
    rows = []
    print("\n================= GENERATIONS =================\n")

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        if input_ids.size(1) > mcfg.seq_len:
            input_ids = input_ids[:, -mcfg.seq_len :]

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=eos_token_id,
            )

        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

        print(f"[PROMPT {i}] {prompt}")
        print(text)
        print("-" * 60)

        rows.append(
            {
                "idx": i,
                "prompt": prompt,
                "generation": text,
                "len_tokens": len(tokenizer.encode(text)),
            }
        )

    
    # W&B logging
    
    if not args.no_wandb:
        table = wandb.Table(
            columns=["idx", "prompt", "generation", "len_tokens"]
        )
        for r in rows:
            table.add_data(
                r["idx"],
                r["prompt"],
                r["generation"],
                r["len_tokens"],
            )

        wandb.log({"generations": table})
        wandb.run.summary["num_generations"] = len(rows)

    print("\n================================================\n")
    wandb.finish()


if __name__ == "__main__":
    main()
