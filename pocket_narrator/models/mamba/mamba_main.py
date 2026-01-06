# pocket_narrator/models/mamba/mamba_main.py
from __future__ import annotations

import argparse
import os
from typing import Dict, Any
import yaml

from pocket_narrator.models.mamba.mamba_model import MambaConfig, MambaLM
from pocket_narrator.models.mamba.mamba_trainer import MambaTrainer, MambaTrainingConfig
from pocket_narrator.models.mamba.mamba_utils import set_seed


'''
python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_2k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_2k/tokenizer.yaml \
  --training_config configs/mamba_tinystories_2k/training.yaml \
  --num_workers 0


python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_4k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_4k/tokenizer.yaml \
  --training_config configs/mamba_tinystories_4k/training.yaml \
  --num_workers 0


python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_8k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_8k/tokenizer.yaml \
  --training_config configs/mamba_tinystories_8k/training.yaml \
  --num_workers 0
  
  python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_10k/model.yaml \
  --tokenizer_config configs/mamba_tinystories_10k/tokenizer.yaml \
  --training_config configs/mamba_tinystories_10k/training.yaml \
  --num_workers 0

python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_config configs/mamba_tinystories_1M/tokenizer.yaml \
  --training_config configs/mamba_tinystories_1M/training.yaml \
  --num_workers 0

  
  '''




def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_mamba_from_config(cfg: dict) -> MambaLM:
    # A) cfg["model"] = {...}
    # B) cfg itself already contains model fields (flat)
    if "model" in cfg and isinstance(cfg["model"], dict):
        m = cfg["model"]
    else:
        m = cfg  # fallback for flat model.yaml

    required = ["vocab_size"]
    missing = [k for k in required if k not in m]
    if missing:
        raise KeyError(
            f"Missing required model keys: {missing}. "
            f"Top-level keys are: {list(cfg.keys())}"
        )
    mcfg = MambaConfig(
        vocab_size=int(m["vocab_size"]),
        d_model=int(m.get("d_model", 512)),
        n_layers=int(m.get("n_layers", 6)),
        seq_len=int(m.get("seq_len", 256)),
        d_state=int(m.get("d_state", 16)),
        d_conv=int(m.get("d_conv", 3)),
        expand=int(m.get("expand", 2)),
        dropout=float(m.get("dropout", 0.1)),
        pad_token_id=int(m.get("pad_token_id", 0)),
        layer_norm_eps=float(m.get("layer_norm_eps", 1e-5)),
    )
    return MambaLM(mcfg)


def build_training_config(cfg: dict) -> MambaTrainingConfig:
    t = cfg["training"] if ("training" in cfg and isinstance(cfg["training"], dict)) else cfg

    required = ["lm_dataset_dir", "output_dir"]
    missing = [k for k in required if k not in t]
    if missing:
        raise KeyError(
            f"Missing required training keys: {missing}. "
            f"Top-level keys are: {list(cfg.keys())}"
        )

    tcfg = MambaTrainingConfig(
        lm_dataset_dir=t["lm_dataset_dir"],
        output_dir=t["output_dir"],
        batch_size=int(t.get("batch_size", 32)),
        eval_batch_size=int(t.get("eval_batch_size", t.get("batch_size", 32))),
        num_epochs=int(t.get("num_epochs", 3)),
        learning_rate=float(t.get("learning_rate", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        grad_accum_steps=int(t.get("grad_accum_steps", 1)),
        device=str(t.get("device", "cuda")),
        seed=int(t.get("seed", 42)),
        log_interval=int(t.get("log_interval", 50)),
        eval_interval=int(t.get("eval_interval", 500)),
    )
    os.makedirs(tcfg.output_dir, exist_ok=True)
    return tcfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--tokenizer_config", type=str, required=False)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg = deep_merge(cfg, load_yaml(args.model_config))
    if args.tokenizer_config:
        cfg = deep_merge(cfg, load_yaml(args.tokenizer_config))
    cfg = deep_merge(cfg, load_yaml(args.training_config))

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    model = build_mamba_from_config(cfg)
    train_cfg = build_training_config(cfg)

    trainer = MambaTrainer(model, train_cfg, num_workers=args.num_workers)
    trainer.train()

    ppl = trainer.eval_perplexity(max_batches=50)
    print(f"Eval Perplexity (subset): {ppl:.2f}")


if __name__ == "__main__":
    main()
