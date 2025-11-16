"""
Defines the abstract base class for all positional encoding methods.
"""
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class AbstractPositionalEncoding(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            torch.Tensor: Output tensor of the same shape with positional info added.
        """
        pass