"""
Contains the implementation of a neural Mamba-based language model.
"""
import os
import json
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .base_model import AbstractLanguageModel

try:
    from mamba_ssm import Mamba  # external library
except ImportError as e:
    raise ImportError(
        "mamba-ssm is not installed. Please run 'python -m pip install --upgrade pip setuptools wheel' "
        " than using  `pip install mamba-ssm` "

        "before using MambaLanguageModel."
    ) from e

@dataclass
class MambaConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0


class MambaLM(nn.Module):
    """
    Simple decoder-only Mamba-based language model.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Stack of Mamba layers
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx:     (B, T) integer token ids
        targets: (B, T) or None

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar or None
        """
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.config.max_seq_len}"
            )

        device = idx.device

        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos_ids = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos_ids)  # (1, T, d_model)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss



# Dataset for next-token prediction

class SequenceDataset(Dataset):
    """
    Builds training samples from tokenized stories.
    From a long token sequence, windows of length <= max_seq_len+1 are taken,
    where the first T tokens are input and the next T are targets.
    """

    def __init__(self, token_sequences: List[List[int]], max_seq_len: int):
        super().__init__()
        self.samples: List[List[int]] = []
        self.max_seq_len = max_seq_len

        for seq in token_sequences:
            if len(seq) < 2:
                continue

            # We cut the sequence into windows.
            start = 0
            while start + 2 <= len(seq):
                end = min(start + max_seq_len + 1, len(seq))
                window = seq[start:end]

                if len(window) < 2:
                    break

                self.samples.append(window)
                if end == len(seq):
                    break
                # simple overlap of 1 token
                start = end - 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window = self.samples[idx]
        window = torch.tensor(window, dtype=torch.long)
        x = window[:-1]
        y = window[1:]
        return x, y


def collate_fn(batch, pad_token_id: int, max_seq_len: int):
    """
    Simple collate function with padding/clipping to max_seq_len.
    """
    xs, ys = zip(*batch)
    max_len = min(max(x.size(0) for x in xs), max_seq_len)

    batch_x = []
    batch_y = []
    for x, y in zip(xs, ys):
        x = x[:max_len]
        y = y[:max_len]
        pad_len = max_len - x.size(0)

        if pad_len > 0:
            pad_x = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_y = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            x = torch.cat([x, pad_x], dim=0)
            y = torch.cat([y, pad_y], dim=0)

        batch_x.append(x)
        batch_y.append(y)

    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)




    # TODO: implement train_step, generate, save, load etc. passend zu AbstractLanguageModel