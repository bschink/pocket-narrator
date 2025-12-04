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
    

# linear attention according to "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (Katharopoulos et al., 2020)
class LinearAttention(AbstractAttention):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1,
                 pos_encoding_module: AbstractPositionalEncoding = None):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding_module
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        The kernel function phi(x) = elu(x) + 1.
        Ensures values are non-negative.
        """
        return F.elu(x) + 1.0

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None, 
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_causal: bool = True
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        batch_size, seq_len, _ = x.shape
        
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        # reshape: (batch, n_head, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # ideally additive embeddings should be used before this layer
        # but if a module is provided, we apply it for compatibility
        if self.pos_encoding is not None:
            offset = 0 
            q = self.pos_encoding(q, offset=offset)
            k = self.pos_encoding(k, offset=offset)

        Q = self.feature_map(q)
        K = self.feature_map(k)
        
        if layer_past is not None:
            # --- INFERENCE MODE (RNN step) ---
            # layer_past contains (S, Z)
            # S: Sum of (phi(K) * V^T) [state matrix] (Batch, Head, D, D)
            # Z: Sum of phi(K) [normalizer] (Batch, Head, D)

            prev_S, prev_Z = layer_past
            
            # K, V, Q are currently (Batch, Head, 1, D)
            k_t = K.squeeze(2) # (B, H, D)
            v_t = v.squeeze(2) # (B, H, D)
            q_t = Q.squeeze(2) # (B, H, D)
            
            # update state: S_t = S_{t-1} + phi(k_t)^T * v_t
            # outer product: (B,H,D,1) @ (B,H,1,D) -> (B,H,D,D)
            kv_outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            
            current_S = prev_S + kv_outer
            current_Z = prev_Z + k_t
            
            # compute output: O_t = (phi(q_t) * S_t) / (phi(q_t) * Z_t)
            # numerator: (B, H, 1, D) @ (B, H, D, D) -> (B, H, 1, D)
            numerator = torch.einsum('bhd,bhde->bhe', q_t, current_S)
            
            # denominator: dot product (B, H, D) . (B, H, D) -> (B, H, 1)
            denominator = (q_t * current_Z).sum(dim=-1, keepdim=True)
            
            y = numerator / (denominator + 1e-6)
            y = y.unsqueeze(2) # restore seq_len=1 dim: (B, H, 1, D)
            
            present = (current_S, current_Z)
            
        else:
            # --- TRAINING MODE (Iterative/Scan) ---
            # memory-efficient implementation that avoids materializing (B, H, L, D, D) tensors
            # formula: O_i = (Q_i @ S_i) / (Q_i @ Z_i), where S_i = sum_{j<=i}(K_j^T V_j)
            
            device = Q.device
            dtype = Q.dtype
            
            # S: running sum of outer products K^T @ V, shape (B, H, D, D)
            # Z: running sum of K, shape (B, H, D)
            S = torch.zeros(batch_size, self.n_head, self.d_k, self.d_k, device=device, dtype=dtype)
            Z = torch.zeros(batch_size, self.n_head, self.d_k, device=device, dtype=dtype)
            
            outputs = []
            
            for t in range(seq_len):
                k_t = K[:, :, t, :]
                v_t = v[:, :, t, :]
                q_t = Q[:, :, t, :]
                
                # update state: S += k_t outer v_t
                # (B,H,D) x (B,H,D) -> (B,H,D,D)
                S = S + torch.einsum('bhd,bhe->bhde', k_t, v_t)
                Z = Z + k_t
                
                # compute output for this position
                # numerator: q_t @ S -> (B, H, D)
                numerator = torch.einsum('bhd,bhde->bhe', q_t, S)
                # denominator: q_t dot Z -> (B, H, 1)
                denominator = (q_t * Z).sum(dim=-1, keepdim=True)
                
                o_t = numerator / (denominator + 1e-6)
                outputs.append(o_t)
            
            # stack outputs: list of (B, H, D) -> (B, H, L, D)
            y = torch.stack(outputs, dim=2)
            
            present = (S, Z)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(y), present