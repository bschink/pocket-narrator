"""
Contains the implementation of a neural Mamba-based language model.

"""
# pocket_narrator/models/mamba/mamba_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pocket_narrator.models.base_model import AbstractLanguageModel


@dataclass
class MambaConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0
    max_new_tokens: int = 20  # for generation in predict_sequence_batch


@dataclass
class MambaOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, conv_kernel_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        padding = conv_kernel_size // 2
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=d_model,  # depthwise
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        residual = x

        x = self.norm(x)
        proj = self.in_proj(x)
        gate, cand = proj.chunk(2, dim=-1)

        cand = cand.transpose(1, 2)  # (B, d, T)
        cand = self.conv(cand)
        cand = cand.transpose(1, 2)  # (B, T, d)

        cand = self.activation(cand)
        gate = torch.sigmoid(gate)

        x = gate * cand
        x = self.out_proj(x)
        x = self.dropout(x)

        return residual + x


class MambaLM(AbstractLanguageModel, nn.Module):
    """
    Mamba Language Model compatible with your AbstractLanguageModel interface.
    """

    def __init__(self, config: MambaConfig):
        AbstractLanguageModel.__init__(self, config.vocab_size)
        nn.Module.__init__(self)

        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [MambaBlock(config.d_model, config.dropout) for _ in range(config.n_layers)]
        )

        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # PyTorch forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        labels: Optional[torch.Tensor] = None,  # (B, T)
    ) -> MambaOutput:
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} > max_seq_len={self.config.max_seq_len}. "
                "Increase max_seq_len in MambaConfig."
            )

        x = self.token_embed(input_ids)  # (B, T, d)

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # teacher-forcing LM loss; ignore pad_token_id
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return MambaOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------------
    # AbstractLanguageModel API
    # ------------------------------------------------------------------

    def predict_sequence_batch(self, input_tokens_batch: list[list[int]]) -> list[list[int]]:
        """
        Simple greedy decoding for demonstration.
        """
        self.eval()
        device = next(self.parameters()).device
        results: list[list[int]] = []

        with torch.no_grad():
            for seq in input_tokens_batch:
                tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
                for _ in range(self.config.max_new_tokens):
                    out = self.forward(tokens)
                    logits = out.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (1,)
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                results.append(tokens[0].tolist())

        return results

    def save(self, model_path: str):
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self.config,
            },
            model_path,
        )

    @classmethod
    def load(cls, model_path: str, config: dict | None = None) -> "MambaLM":
        checkpoint = torch.load(model_path, map_location="cpu")
        # Prefer config from checkpoint; fall back to passed dict if needed
        if isinstance(checkpoint.get("config"), MambaConfig):
            cfg = checkpoint["config"]
        elif config is not None:
            cfg = MambaConfig(**config)
        else:
            raise ValueError("No valid MambaConfig found in checkpoint and no config provided.")

        model = cls(cfg)
        model.load_state_dict(checkpoint["model_state"])
        return model








'''

try:
    from mamba_ssm import Mamba  # external library
except ImportError as e:
    raise ImportError(
        "mamba-ssm is not installed. Please run 'python -m pip install --upgrade pip setuptools wheel' "
        " than using  `pip install mamba-ssm` "

        "before using MambaLanguageModel."
    ) from e
# pocket_narrator/models/mamba/mamba_model.py



@dataclass
class MambaConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 256


class MambaBlock(nn.Module):
    """
    Vereinfachter, Mamba-inspirierter Block:
    - LayerNorm
    - lineare Projektion -> 2 * d_model
    - gated depthwise-Conv + nichtlineare Aktivierung
    - Residual-Verknüpfung
    Kein originaler CUDA-Kernel, aber O(T * d_model) und gut für Mac.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, conv_kernel_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # Projektion in 2 * d_model: gate + candidate
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # Depthwise-Conv über die Zeitdimension (T)
        padding = conv_kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding=padding,
            groups=d_model,  # depthwise
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # smooth ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # (B, T, 2*d)
        x_proj = self.in_proj(x)
        gate, candidate = x_proj.chunk(2, dim=-1)  # (B, T, d)

        # depthwise Conv arbeitet auf (B, C, T)
        candidate = candidate.transpose(1, 2)      # (B, d, T)
        candidate = self.conv(candidate)
        candidate = candidate.transpose(1, 2)      # (B, T, d)

        candidate = self.activation(candidate)
        gate = torch.sigmoid(gate)

        x = gate * candidate
        x = self.out_proj(x)
        x = self.dropout(x)

        return residual + x  # Residualpfad
        

class MambaLM(AbstractLanguageModel):
    """
    Ein LM mit:
    - Token-Embedding
    - Positions-Embedding
    - Stack von vereinfachten Mamba-Blöcken
    - LayerNorm + Output-Projektion auf Vokabelgröße
    """

    def __init__(self, config: MambaConfig):
        super().__init__(config.vocab_size)
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [MambaBlock(config.d_model, config.dropout) for _ in range(config.n_layers)]
        )

        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d)
        """
        B, T, _ = x.size()
        device = x.device
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embed(positions)                      # (1, T, d)
        return x + pos_emb

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,           # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> LMOutput:
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} > max_seq_len {self.config.max_seq_len}. "
                "Bitte max_seq_len im Config erhöhen."
            )

        x = self.token_embed(input_ids)        # (B, T, d)
        x = self._add_positional_encoding(x)   # (B, T, d)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)                       # (B, T, d)

        x = self.norm_f(x)
        logits = self.lm_head(x)               # (B, T, V)

        loss = None
        if labels is not None:
            # Standard LM-CE-Loss über (B*T, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return LMOutput(logits=logits, loss=loss)


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
    pad_token_id: int = None  # Must be set before training


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


'''

