"""
Defines the abstract base class for all trainers.
"""
from abc import ABC, abstractmethod
from pocket_narrator.models.base_model import AbstractLanguageModel

class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, model: AbstractLanguageModel, tokenizer, train_data: list[str]) -> AbstractLanguageModel:
        """
        The main training method.

        Args:
            model: The model instance to be trained.
            tokenizer: The tokenizer instance.
            train_data: The list of training sentences.

        Returns:
            The trained model instance.
        """
        pass