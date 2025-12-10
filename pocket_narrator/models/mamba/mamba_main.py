# pocket_narrator/models/mamba/mamba_main.py

from __future__ import annotations

import argparse
import os

import torch
import yaml

from pocket_narrator.models.mamba.mamba_model import MambaConfig, MambaLM
from pocket_narrator.models.mamba.mamba_trainer import (
    MambaTrainer,
    MambaTrainingConfig,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mamba_from_config(cfg: dict) -> MambaLM:
    mcfg = MambaConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"].get("d_model", 512),
        n_layers=cfg["model"].get("n_layers", 6),
        max_seq_len=cfg["model"].get("max_seq_len", 256),
        dropout=cfg["model"].get("dropout", 0.1),
        pad_token_id=cfg["model"].get("pad_token_id", 0),
        max_new_tokens=cfg["model"].get("max_new_tokens", 20),
    )
    return MambaLM(mcfg)


def build_training_config(cfg: dict) -> MambaTrainingConfig:
    tcfg = MambaTrainingConfig(
        lm_dataset_dir=cfg["training"]["lm_dataset_dir"],
        output_dir=cfg["training"]["output_dir"],
        batch_size=cfg["training"].get("batch_size", 32),
        eval_batch_size=cfg["training"].get("eval_batch_size", 32),
        num_epochs=cfg["training"].get("num_epochs", 3),
        learning_rate=cfg["training"].get("learning_rate", 3e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
        device=cfg["training"].get("device", "cuda"),
        log_interval=cfg["training"].get("log_interval", 100),
        eval_interval=cfg["training"].get("eval_interval", 1000),
        seed=cfg["training"].get("seed", 42),
    )
    os.makedirs(tcfg.output_dir, exist_ok=True)
    return tcfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config for Mamba model & training.",
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml(args.config)
        model = build_mamba_from_config(cfg)
        train_cfg = build_training_config(cfg)
        trainer = MambaTrainer(model, train_cfg)

        trainer.train()
        ppl = trainer.eval_perplexity(max_batches=50)
        print(f"Eval Perplexity (subset): {ppl:.2f}")
    else:
        # Fallback: simple smoke test without YAML
        print("No config provided, running simple smoke test...")
        config = MambaConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            dropout=0.1,
            max_seq_len=256,
        )
        model = MambaLM(config)

        B, T = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels)
        print("Logits shape:", output.logits.shape)
        print("Loss:", output.loss.item() if output.loss is not None else None)


if __name__ == "__main__":
    main()
