# pocket_narrator/models/mamba/mamba_trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_from_disk

from pocket_narrator.models.mamba.mamba_model import MambaLM
from pocket_narrator.models.mamba.mamba_utils import set_seed, HFDatasetWrapper


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
    device: str = "cuda"
    log_interval: int = 100
    eval_interval: int = 1000
    seed: int = 42


class MambaTrainer:
    def __init__(
        self,
        model: MambaLM,
        train_config: MambaTrainingConfig,
    ):
        self.model = model
        self.cfg = train_config

        set_seed(self.cfg.seed)

        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Dataset laden
        raw_ds = load_from_disk(self.cfg.lm_dataset_dir)

        # Falls DatasetDict mit 'train'-Split, sonst direkt Dataset
        if isinstance(raw_ds, dict) and "train" in raw_ds:
            train_ds = raw_ds["train"]
        else:
            train_ds = raw_ds

        self.train_loader = DataLoader(
            HFDatasetWrapper(train_ds),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def train(self):
        global_step = 0
        self.model.train()

        for epoch in range(self.cfg.num_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Model returned no loss during training.")

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                global_step += 1

                if global_step % self.cfg.log_interval == 0:
                    pbar.set_postfix({"loss": loss.item()})

            # nach jeder Epoche Modell speichern
            save_path = f"{self.cfg.output_dir}/mamba_epoch{epoch+1}.pt"
            self.model.save(save_path)

    @torch.no_grad()
    def eval_perplexity(self, max_batches: Optional[int] = None) -> float:
        """
        Grobe Perplexity auf dem Trainingsdataset (oder eval split, falls du es anpasst).
        """
        self.model.eval()

        ds = load_from_disk(self.cfg.lm_dataset_dir)
        if isinstance(ds, dict) and "validation" in ds:
            ds_eval = ds["validation"]
        else:
            ds_eval = ds

        loader = DataLoader(
            HFDatasetWrapper(ds_eval),
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
        )

        n_tokens = 0
        total_loss = 0.0

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            if loss is None:
                continue

            n_tok = (batch["labels"] != self.model.config.pad_token_id).sum().item()
            total_loss += loss.item() * n_tok
            n_tokens += n_tok

        mean_loss = total_loss / max(n_tokens, 1)
        perplexity = float(torch.exp(torch.tensor(mean_loss)))
        return perplexity
