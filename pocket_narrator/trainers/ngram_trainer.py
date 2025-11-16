"""
Contains the training logic specific to the NGramModel.
"""
from tqdm import tqdm
from .base_trainer import AbstractTrainer
from pocket_narrator.models.base_model import AbstractLanguageModel

class NGramTrainer(AbstractTrainer):
    def train(self, model: AbstractLanguageModel, tokenizer, train_data: list[str]) -> AbstractLanguageModel:
        """
        Trains an n-gram model by feeding it the entire tokenized training corpus.
        """
        print("--- Running NGramTrainer ---")
        print(f"INFO: Tokenizing {len(train_data)} training samples...")
        all_train_tokens = [tokenizer.encode(text) for text in tqdm(train_data, desc="Tokenizing", unit="sample")]
        model.train(all_train_tokens)
        return model