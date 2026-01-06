from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_from_disk, DatasetDict

from pocket_narrator.models.mamba.mamba_model import MambaLM


class LMDataset(torch.utils.data.Dataset):
    """
    Expects a HF dataset on disk with column 'input_ids'.
    Each row is a fixed-length token sequence.
    """

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ids = self.ds[idx]["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def create_dataloaders(
    lm_dataset_dir: str,
    train_batch_size: int,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a HF dataset (saved with `save_to_disk`) and wrap in PyTorch DataLoaders.

    Handles both:
    - DatasetDict with splits (e.g. {"train": ..., "validation": ...})
    - Single Dataset (no splits)
    """
    ds = load_from_disk(lm_dataset_dir)

    if isinstance(ds, DatasetDict):
        train_ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        eval_ds = ds["validation"] if "validation" in ds else train_ds
    else:
        train_ds = ds
        eval_ds = ds

    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    use_pin = torch.cuda.is_available()

    train_dl = DataLoader(
        LMDataset(train_ds),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    eval_dl = DataLoader(
        LMDataset(eval_ds),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_dl, eval_dl


def _check_token_range(input_ids: torch.Tensor, vocab_size: int) -> None:
    mx = int(input_ids.max().item())
    mn = int(input_ids.min().item())

    if mn < 0:
        raise ValueError(
            f"Found negative token id {mn}. Something is wrong with dataset/tokenizer."
        )

    if mx >= vocab_size:
        raise ValueError(
            f"Token id out of range: max_id={mx} but vocab_size={vocab_size}. "
            f"LM dataset + tokenizer do NOT match the model vocab."
        )


def train_one_epoch(
    model: MambaLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _check_token_range(input_ids, model.config.vocab_size)

        out = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = out["loss"] / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model: MambaLM, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _check_token_range(input_ids, model.config.vocab_size)

        out = model(input_ids=input_ids, labels=labels, return_dict=True)
        total_loss += float(out["loss"].item())
        steps += 1

    avg_loss = total_loss / max(steps, 1)

    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return avg_loss, ppl


@dataclass
class MambaTrainingConfig:
    lm_dataset_dir: str
    output_dir: str

    batch_size: int = 32
    eval_batch_size: int = 32
    num_epochs: int = 3

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1

    device: str = "cuda"
    seed: int = 42

    log_interval: int = 50
    eval_interval: int = 500


class MambaTrainer:
    def __init__(self, model: MambaLM, cfg: MambaTrainingConfig, num_workers: int = 0):
        self.model = model
        self.cfg = cfg
        self.num_workers = num_workers

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(cfg.device if (cfg.device == "cpu" or use_cuda) else "cpu")
        self.model.to(self.device)

        os.makedirs(cfg.output_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.train_dl, self.eval_dl = create_dataloaders(
            lm_dataset_dir=cfg.lm_dataset_dir,
            train_batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size,
            num_workers=num_workers,
        )

    def save_checkpoint(self, name: str = "checkpoint.pt"):
        path = Path(self.cfg.output_dir) / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_config": self.cfg.__dict__,
                "model_config": getattr(self.model, "config", None),
            },
            path,
        )

    def train(self):
        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_dl,
                optimizer=self.optimizer,
                device=self.device,
                grad_accum_steps=self.cfg.grad_accum_steps,
                max_grad_norm=self.cfg.max_grad_norm,
            )

            eval_loss, eval_ppl = evaluate(self.model, self.eval_dl, self.device)

            print(
                f"[EPOCH {epoch}/{self.cfg.num_epochs}] "
                f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
            )

            self.save_checkpoint(f"epoch_{epoch}.pt")

    @torch.no_grad()
    def eval_perplexity(self, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in self.eval_dl:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _check_token_range(input_ids, self.model.config.vocab_size)

            out = self.model(input_ids=input_ids, labels=labels, return_dict=True)
            total_loss += float(out["loss"].item())
            steps += 1

            if steps >= max_batches:
                break

        avg_loss = total_loss / max(steps, 1)
        try:
            return math.exp(avg_loss)
        except OverflowError:
            return float("inf")
