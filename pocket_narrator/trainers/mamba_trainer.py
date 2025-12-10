
from __future__ import annotations
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from pocket_narrator.models.mamba_model import SequenceDataset, collate_fn
from pocket_narrator.models.base_model import AbstractLanguageModel
from .base_trainer import AbstractTrainer



class MambaTrainer(AbstractTrainer):

    def train(
        self,
        train_tokens: List[List[int]],
        batch_size: int = 32,
        epochs: int = 3,
        lr: float = 3e-4,
    ):
        """
        Trains the Mamba LM on the given tokenized stories. Args:
        rain_tokens: List of lists of token IDs.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        """
        print("INFO: Training Mamba language model...")
        dataset = SequenceDataset(
            token_sequences=train_tokens,
            max_seq_len=self.config.max_seq_len,
        )

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(
                b, self.config.pad_token_id, self.config.max_seq_len
            ),
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in tqdm(dl, desc=f"Epoch {epoch}", unit="batch"):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                _, loss = self.model(x, targets=y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(dl))
            print(f"INFO: Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")

    # Prediction
    @torch.no_grad()
    def predict_sequence_batch(
        self,
        input_tokens_batch: List[List[int]],
        max_length: int = 400,
        strategy: str = "greedy",
        no_repeat_ngram_size: int = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Generates continuations for each prompt sequence in the batch.
        - strategy: 'greedy' or 'sample'
        - no_repeat_ngram_size: if set (e.g., 3), attempts are made to
        avoid repeating n-grams of this size.
        """
        self.model.eval()
        predictions = []

        for prompt_tokens in input_tokens_batch:
            generated = list(prompt_tokens)

            while len(generated) < len(prompt_tokens) + max_length:
                # Kontext auf max_seq_len beschränken
                context = generated[-self.config.max_seq_len :]
                context_tensor = torch.tensor(
                    context, dtype=torch.long, device=self.device
                ).unsqueeze(0)  # (1, T)

                logits, _ = self.model(context_tensor)
                next_logits = logits[0, -1, :]  # (vocab_size,)

                # Temperatur-Skalierung
                if temperature is not None and temperature > 0:
                    next_logits = next_logits / temperature

                # optional top-k
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(next_logits, k=top_k)
                    min_logit = values[-1]
                    next_logits[next_logits < min_logit] = -float("inf")

                probs = F.fix.softmax(next_logits, dim=-1)

                # optional no-repeat-ngram-Filter
                if no_repeat_ngram_size is not None:
                    probs_np = probs.clone()
                    for token_id in range(self.vocab_size):
                        if self._would_create_repeated_ngram(
                            generated,
                            token_id,
                            no_repeat_ngram_size,
                        ):
                            probs_np[token_id] = 0.0
                    if probs_np.sum() > 0:
                        probs = probs_np / probs_np.sum()

                if strategy == "sample":
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    # greedy
                    next_token = int(torch.argmax(probs).item())

                if next_token == self.eos_token_id:
                    break

                generated.append(next_token)

            predictions.append(generated[len(prompt_tokens):])

        return predictions

    # Save 

    def save(self, model_path: str):
        """
        Stores weights and configuration of the Mamba model.
        """
        print(f"INFO: Saving Mamba model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        save_obj = {
            "config": {
                "model_type": "mamba",
                "vocab_size": self.vocab_size,
                "eos_token_id": self.eos_token_id,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "d_state": self.config.d_state,
                "d_conv": self.config.d_conv,
                "expand": self.config.expand,
                "max_seq_len": self.config.max_seq_len,
                "dropout": self.config.dropout,
                "pad_token_id": self.config.pad_token_id,
            },
            "state_dict": self.model.state_dict(),
        }

        torch.save(save_obj, model_path) # pyright: ignore[reportUndefinedVariable]

    @classmethod
    def load(cls, model_path: str, config: dict):
        """
        Lädt ein Mamba-Modell aus Datei.

        Args:
            model_path: Pfad zur .pt/.bin-Datei (torch.save).
            config: Dictionary mit Hyperparametern, z.B. aus JSON/CLI.
        """
        print("INFO: Instantiating Mamba model from config...")

        vocab_size = int(config["vocab_size"])
        eos_token_id = int(config["eos_token_id"])

        model = cls(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            d_model=int(config.get("d_model", 512)),
            n_layers=int(config.get("n_layers", 4)),
            d_state=int(config.get("d_state", 16)),
            d_conv=int(config.get("d_conv", 4)),
            expand=int(config.get("expand", 2)),
            max_seq_len=int(config.get("max_seq_len", 256)),
            dropout=float(config.get("dropout", 0.1)),
            pad_token_id=int(config.get("pad_token_id", -100)),
        )

        saved = torch.load(model_path, map_location=model.device)
        state_dict = saved.get("state_dict", saved)
        model.model.load_state_dict(state_dict)

        return model