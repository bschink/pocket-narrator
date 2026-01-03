# pocket_narrator/models/mamba/mamba_trainer.py

import math
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_from_disk, Dataset, DatasetDict

from pocket_narrator.models.mamba.mamba_model import MambaLM, MambaConfig


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
    batch_size: int,
    num_workers: int = 4,
):
    """
    Load a HF dataset (saved with `save_to_disk`) and wrap in PyTorch DataLoaders.

    Handles both:
    - DatasetDict with splits (e.g. {"train": ..., "validation": ...})
    - Single Dataset (no splits)
    """
    ds = load_from_disk(lm_dataset_dir)

    # ds can be a DatasetDict (with splits) or a single Dataset
    if isinstance(ds, DatasetDict):
        # Prefer 'train' if it exists
        if "train" in ds:
            train_ds = ds["train"]
        else:
            first_key = list(ds.keys())[0]
            train_ds = ds[first_key]

        # Prefer 'validation' if it exists; else evaluate on train
        if "validation" in ds:
            eval_ds = ds["validation"]
        else:
            eval_ds = train_ds
    else:
        # Single Dataset → use same for train and eval
        train_ds = ds
        eval_ds = ds

    use_pin = (torch.cuda.is_available()) # only pin on CUDA

    train_dl = DataLoader(
        LMDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    eval_dl = DataLoader(
        LMDataset(eval_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    return train_dl, eval_dl


def train_one_epoch(
    model: MambaLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
):
    """
    One training epoch over the dataloader with gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # right before: out = model(...)
        mx = int(input_ids.max().item())
        mn = int(input_ids.min().item())

        if mn < 0:
            raise ValueError(f"Found negative token id {mn}. Something is wrong with dataset/tokenizer.")

        # model.vocab_size depends on how you stored it; adjust if needed
        vocab_size = model.mcfg.vocab_size if hasattr(model, "mcfg") else model.config.vocab_size

        if mx >= vocab_size:
            raise ValueError(
                f"Token id out of range: max_id={mx} but vocab_size={vocab_size}. "
                f"Your LM dataset + tokenizer do NOT match the model vocab."
            )

        out = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = out["loss"] / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(
    model: MambaLM,
    dataloader: DataLoader,
    device: torch.device,
):
    """
    Evaluate loss and perplexity on the given dataloader.
    """
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = out["loss"]
        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(steps, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl
