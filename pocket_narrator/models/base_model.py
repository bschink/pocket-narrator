"""
Defines the abstract base class for all models in the project.
"""
from abc import ABC, abstractmethod

class AbstractLanguageModel(ABC):
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        super().__init__()
    
    @abstractmethod
    def predict_sequence_batch(self, input_tokens_batch: list[list[int]]) -> list[list[int]]:
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, config: dict):
        pass