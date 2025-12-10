"""
Implements a single, decoder-only Transformer block.
Supports configurable FFN activations (GELU vs SwiGLU) with 
automatic parameter matching logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_attention import AbstractAttention

class FeedForward(nn.Module):
    """
    Feed-Forward Network that supports 'gelu' and 'swiglu'.
    
    Efficiency Note:
    - If 'swiglu' is selected, we automatically scale down the hidden dimension 
      by 2/3. This ensures that the SwiGLU FFN has roughly the same number of 
      parameters as a standard GELU FFN with expansion_factor=4.
    """
    def __init__(self, d_model: int, expansion_factor: int, dropout: float, activation_type: str):
        super().__init__()
        self.activation_type = activation_type.lower()
        self.dropout = nn.Dropout(dropout)

        if self.activation_type == "gelu":
            # Standard Transformer FFN: d -> 4d -> d
            d_hidden = d_model * expansion_factor
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
                self.dropout
            )
        elif self.activation_type == "swiglu":
            # SwiGLU: (Swish(xW_g) * xW_v) * W_o
            # To match parameter count of GELU (2 matrices vs 3 matrices),
            # we scale hidden dim by 2/3.
            # 2 * (4*d^2) = 8d^2  (GELU cost)
            # 3 * (h*d) = 8d^2 => h = (8/3)d = 4d * (2/3)
            d_hidden = int(d_model * expansion_factor * 2 / 3)
            
            if d_hidden % 2 != 0:
                d_hidden += 1

            self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
            self.w_val = nn.Linear(d_model, d_hidden, bias=False)
            self.w_out = nn.Linear(d_hidden, d_model, bias=False)
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

    def forward(self, x):
        if self.activation_type == "gelu":
            return self.net(x)
        elif self.activation_type == "swiglu":
            # SiLU is Swish with beta=1
            gate = F.silu(self.w_gate(x))
            val = self.w_val(x)
            out = self.w_out(gate * val)
            return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, attention_module: AbstractAttention, dropout: float, activation_type: str = "gelu"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = attention_module
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, 
            expansion_factor=4, 
            dropout=dropout, 
            activation_type=activation_type
        ) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, 
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_causal: bool = False):
        # using Pre-LayerNorm architecture for better training stability  
        # Attention Sub-layer
        attn_out, present = self.attn(self.ln_1(x), mask=mask, layer_past=layer_past, is_causal=is_causal)
        x = x + self.dropout(attn_out)
        
        # Feed-Forward Sub-layer
        x = x + self.ffn(self.ln_2(x))
        
        return x, present