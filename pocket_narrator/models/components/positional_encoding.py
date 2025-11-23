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

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        x shape: (batch_size, seq_len, d_model)
        offset: The starting position index (used for KV-Caching inference)
        """
        seq_len = x.size(1)
        pe_slice = self.pe[:, offset : offset + seq_len, :]
        
        x = x + pe_slice
        return self.dropout(x)
    
class RotaryPositionalEncoding(AbstractPositionalEncoding):
    """
    Implements Rotary Positional Embeddings (RoPE) from the paper RoFormer.
    https://arxiv.org/abs/2104.09864

    This module does not add to the input tensor but is instead applied to the
    Query and Key tensors within the attention mechanism.
    """
    def __init__(self, d_model: int, max_len: int, base: int = 10000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for Rotary Positional Encoding."
        
        # compute theta frequencies for rotations
        # shape: (d_model / 2)
        theta = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))

        # compute frequency map for all positions creating matrix of all m * theta_i values.
        # shape: (max_len, d_model / 2)
        seq_idx = torch.arange(max_len)
        freqs = torch.outer(seq_idx, theta)

        # convert to complex numbers
        # multiplication with a complex number of magnitude 1 (e^i*theta) -> rotation representation
        # shape: (max_len, d_model / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        # register as buffer so it moves to correct device with model
        # shape: (1, max_len, 1, d_model / 2)
        self.register_buffer('freqs_complex', freqs_complex.unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Applies the rotary rotation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor (Query or Key) of shape 
                              (batch, n_head, seq_len, d_k).
            offset: The starting position index.
        
        Returns:
            torch.Tensor: Rotated tensor of the same shape.
        """
        # reshape x to view the last dimension as pairs of two
        # shape: (batch, n_head, seq_len, d_k) -> (batch, n_head, seq_len, d_k/2, 2)
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        # now as complex numbers
        # shape: (batch, n_head, seq_len, d_k/2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        seq_len = x.shape[2]
        # Slice frequencies from 'offset' to 'offset + seq_len'
        # shape of freqs_complex buffer: (1, max_len, 1, d_k/2)
        # After slicing: (1, seq_len, 1, d_k/2)
        # Transpose to: (1, 1, seq_len, d_k/2)
        freqs = self.freqs_complex[:, offset : offset + seq_len, :, :].transpose(1, 2)
        
        # perform the rotation via element-wise complex multiplication
        # shape of x_complex: (batch, n_head, seq_len, d_k/2)
        # shape of freqs:     (1,     1,      seq_len, d_k/2)
        x_rotated_complex = x_complex * freqs

        # convert back to a real tensor.
        # shape: (batch, n_head, seq_len, d_k/2, 2)
        x_rotated = torch.view_as_real(x_rotated_complex)
        # shape: (batch, n_head, seq_len, d_k)
        x_out = x_rotated.reshape(*x.shape)
        
        return x_out.type_as(x)