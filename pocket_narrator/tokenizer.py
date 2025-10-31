"""
This module contains tokenizer implementations and a factory function
to manage their lifecycle (training, loading, instantiation).
"""
import os
import json
from pathlib import Path

class CharacterTokenizer:
    """A character-level tokenizer that learns its vocabulary from data."""

    def __init__(self):
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.vocabulary = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token_id = None

    def get_vocab_size(self) -> int:
        return len(self.vocabulary)
    
    def train(self, corpus: list[str]):
        print("INFO: Training CharacterTokenizer from corpus...")
        unique_chars = sorted(list(set(''.join(corpus))))
        self.vocabulary = self.special_tokens + unique_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        self.unk_token_id = self.char_to_idx['<unk>']
        print(f"INFO: Vocabulary built. Size: {self.get_vocab_size()} tokens.")

    def save(self, save_path: str):
        """Saves the tokenizer's vocabulary to a single JSON file."""
        if not self.vocabulary:
            raise ValueError("Cannot save an untrained tokenizer.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
        print(f"INFO: Tokenizer vocabulary saved to {save_path}")

    @classmethod
    def load(cls, load_path: str):
        """Loads a tokenizer's vocabulary from a single JSON file."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer vocabulary not found at {load_path}")
        tokenizer = cls()
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer.vocabulary = json.load(f)
        tokenizer.char_to_idx = {char: idx for idx, char in enumerate(tokenizer.vocabulary)}
        tokenizer.idx_to_char = {idx: char for idx, char in enumerate(tokenizer.vocabulary)}
        tokenizer.unk_token_id = tokenizer.char_to_idx.get('<unk>')
        print(f"INFO: Tokenizer loaded from {load_path}. Vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer

    def encode(self, text: str) -> list[int]:
        if not self.vocabulary:
            raise RuntimeError("Tokenizer has not been trained. Call .train() first.")
        return [self.char_to_idx.get(char, self.unk_token_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        if not self.vocabulary:
            raise RuntimeError("Tokenizer has not been trained.")
        return "".join([self.idx_to_char.get(idx, '') for idx in token_ids])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        return [self.decode(tokens) for tokens in token_lists]


def get_tokenizer(tokenizer_type: str = "simple"):
    """
    Factory function to get a tokenizer instance.
    This is the single entry point for the rest of the application.

    Args:
        tokenizer_type (str): The type of tokenizer to return.

    Returns:
        An initialized tokenizer instance.
    """
    if tokenizer_type == "simple":
        print("INFO: Using SimpleCharacterTokenizer.")
        return SimpleTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: '{tokenizer_type}'")
    

def get_tokenizer(
    tokenizer_type: str, 
    tokenizer_path: str = None, 
    train_corpus: list[str] = None
):
    """
    Factory function to manage the lifecycle of a tokenizer.

    - If tokenizer_path exists, it loads a pre-trained tokenizer.
    - If it doesn't exist, it trains a new one using train_corpus and saves it.

    Args:
        tokenizer_type (str): The type of tokenizer ('character').
        tokenizer_path (str, optional): Path to save/load the tokenizer's vocab file.
        train_corpus (list[str], optional): Corpus to train on if tokenizer is new.

    Returns:
        An initialized tokenizer instance.
    """
    if tokenizer_type == "character":
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"INFO: Loading existing CharacterTokenizer from {tokenizer_path}.")
            return CharacterTokenizer.load(tokenizer_path)
        elif train_corpus:
            print(f"INFO: No existing tokenizer found. Training a new CharacterTokenizer.")
            tokenizer = CharacterTokenizer()
            tokenizer.train(train_corpus)
            if tokenizer_path:
                tokenizer.save(tokenizer_path)
            return tokenizer
        else:
            raise ValueError("Must provide either a valid tokenizer_path or a train_corpus for CharacterTokenizer.")
    else:
        raise ValueError(f"Unknown tokenizer type: '{tokenizer_type}'")