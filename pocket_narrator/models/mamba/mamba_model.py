import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    seq_len: int = 256
    d_state: int = 16
    d_conv: int = 3
    expand: int = 2
    pad_token_id: int = 0
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        if d_conv % 2 == 0:
            raise ValueError(
                f"d_conv={d_conv} is even. Kernel size must be odd (3,5,7,...) to preserve seq length."
            )

        padding = d_conv // 2

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=padding,
            groups=d_model,
        )

        inner_dim = d_model * expand
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)

        u_and_gate = self.in_proj(x_conv)
        u, gate = u_and_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        x_mixed = F.silu(u) * gate

        out = self.out_proj(x_mixed)
        return residual + out


class MambaLM(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [
                SimpleMambaBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model maximum {self.config.seq_len}")

        device = input_ids.device
        pos = torch.arange(seq_len, device=device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        if not return_dict:
            return logits if loss is None else (loss, logits)

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_ngram: Optional[int] = 4,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        generated = input_ids

        for _ in range(max_new_tokens):
            if generated.size(1) > self.config.seq_len:
                generated = generated[:, -self.config.seq_len :]

            out = self(generated, labels=None, return_dict=True)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                vals, idx = torch.topk(logits, top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, idx, vals)
                probs = torch.softmax(masked, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            if repetition_ngram is not None and generated.size(1) >= repetition_ngram:
                n = repetition_ngram
                tokens_list = generated.tolist()[0]
                prev = tokens_list[-n + 1 :]
                candidate = prev + [int(next_token.item())]
                for i in range(len(tokens_list) - n + 1):
                    if tokens_list[i : i + n] == candidate:
                        next_token = torch.randint(0, self.config.vocab_size, (1, 1), device=device)
                        break

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break

        return generated
