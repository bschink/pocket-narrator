"""
Contains the training logic specific to the NGramModel.
"""
from .base_trainer import AbstractTrainer
from pocket_narrator.models.base_model import AbstractLanguageModel

class NGramTrainer(AbstractTrainer):
    def train(self, model: AbstractLanguageModel, tokenizer, train_data: list[str]) -> AbstractLanguageModel:
        """
        Trains an n-gram model by feeding it the entire tokenized training corpus.
        """
        print("--- Running NGramTrainer ---")
        all_train_tokens = tokenizer.encode_batch(train_data)
        model.train(all_train_tokens)
        return model