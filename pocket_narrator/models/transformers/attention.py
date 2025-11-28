"""
Contains concrete implementations of attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_attention import AbstractAttention
from ..components.base_pos_encoding import AbstractPositionalEncoding

class MultiHeadSelfAttention(AbstractAttention):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1,
                 pos_encoding_module: AbstractPositionalEncoding = None):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_k = d_model // n_head
        self.n_head = n_head
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoding = pos_encoding_module

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_causal: bool = False
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        
        # project to Q, K, V and reshape for multi-head
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, head_dim)

        # apply RoPE if it exists
        if self.pos_encoding is not None:
            offset = layer_past[0].size(2) if layer_past is not None else 0
            q = self.pos_encoding(q, offset=offset)
            k = self.pos_encoding(k, offset=offset)

        # KV caching logic
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)
        
        # apply scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V
        effective_is_causal = is_causal and mask is None
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask if not is_causal else None, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=effective_is_causal
        )
        
        # concatenate heads and project back to d_model
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(y), present