"""
Defines the abstract base class for all attention mechanisms.
"""
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class AbstractAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        pass