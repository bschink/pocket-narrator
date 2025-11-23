"""
Defines the abstract base class for all attention mechanisms.
"""
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Optional, Tuple

class AbstractAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask.
            layer_past (Tuple[Tensor, Tensor], optional): Cached Key/Value tensors from previous steps.
        
        Returns:
            output (torch.Tensor): The attention output.
            present (Tuple[Tensor, Tensor]): The updated Key/Value tensors to be cached.
        """
        pass