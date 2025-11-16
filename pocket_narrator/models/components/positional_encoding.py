"""
Contains concrete implementations of positional encoding methods.
"""
import torch
import torch.nn as nn
import math
from .base_pos_encoding import AbstractPositionalEncoding

class SinusoidalPositionalEncoding(AbstractPositionalEncoding):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal positional encoding."
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        x shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)