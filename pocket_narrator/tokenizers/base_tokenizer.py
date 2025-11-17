"""
Defines the abstract base class for all tokenizers.
"""
from abc import ABC, abstractmethod
from typing import List, Iterator

class AbstractTokenizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, corpus: List[str]):
        """Trains the tokenizer from a corpus of text."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Returns the total size of the vocabulary."""
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Converts a single string to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs back to a single string."""
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Applies encoding to a batch (list) of texts."""
        pass

    @abstractmethod
    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Applies decoding to a batch (list) of token lists."""
        pass
    
    @abstractmethod
    def save(self, save_path: str):
        """Saves the tokenizer's state to a file or directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_path: str):
        """Loads a tokenizer's state from a file or directory."""
        pass