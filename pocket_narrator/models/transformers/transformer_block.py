"""
Implements a single, decoder-only Transformer block.
"""
import torch.nn as nn
from .base_attention import AbstractAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, attention_module: AbstractAttention, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = attention_module
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # using Pre-LayerNorm architecture for better training stability
        x = x + self.dropout(self.attn(self.ln_1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x