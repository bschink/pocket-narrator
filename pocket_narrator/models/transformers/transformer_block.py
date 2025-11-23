"""
Implements a single, decoder-only Transformer block.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
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

    def forward(self, x, mask=None, 
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # using Pre-LayerNorm architecture for better training stability  
        # Attention Sub-layer
        attn_out, present = self.attn(self.ln_1(x), mask=mask, layer_past=layer_past)
        x = x + self.dropout(attn_out)
        
        # Feed-Forward Sub-layer
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        
        return x, present