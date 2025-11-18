"""
Contains the implementation of a character-level tokenizer that learns
its vocabulary from a training corpus.
"""
import os
import json
from .base_tokenizer import AbstractTokenizer

class CharacterTokenizer(AbstractTokenizer):
    """A character-level tokenizer that learns its vocabulary from data."""

    def __init__(self, special_tokens: list[str] = None):
        self.special_token_names = special_tokens if special_tokens else []
        self.vocabulary = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token_id = None
    
    @property
    def special_tokens(self) -> dict:
        """Return special tokens as a dict mapping token names to IDs."""
        return {token: self.char_to_idx[token] for token in self.special_token_names if token in self.char_to_idx}

    def get_vocab_size(self) -> int:
        return len(self.vocabulary)
    
    def token_to_id(self, token: str) -> int:
        return self.char_to_idx.get(token, self.unk_token_id)
    
    def train(self, corpus: list[str]):
        print("INFO: Training CharacterTokenizer from corpus...")
        unique_chars = sorted(list(set(''.join(corpus))))
        self.vocabulary = self.special_token_names + unique_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        # Set unk_token_id if <unk> exists in vocabulary, otherwise use 0
        self.unk_token_id = self.char_to_idx.get('<unk>', 0)
        print(f"INFO: Vocabulary built. Size: {self.get_vocab_size()} tokens.")

    def save(self, save_path: str):
        """
        Saves the tokenizer's vocabulary to a specified directory.

        Args:
            save_path (str): The path to the DIRECTORY where the vocab file will be saved.
        """
        if not self.vocabulary:
            raise ValueError("Cannot save an untrained tokenizer.")
        
        print(f"INFO: Saving Character tokenizer to directory: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        file_path = os.path.join(save_path, "vocab.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, load_path: str):
        """
        Loads a tokenizer's vocabulary from a directory.

        Args:
            load_path (str): The path to the DIRECTORY containing the vocab.json file.
        """
        print(f"INFO: Loading Character tokenizer from directory: {load_path}")
        file_path = os.path.join(load_path, "vocab.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer vocabulary not found at {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        
        special_tokens_list = []
        for token in vocabulary:
            if token.startswith('<') and token.endswith('>'):
                special_tokens_list.append(token)
        
        tokenizer = cls(special_tokens=special_tokens_list)
            
        tokenizer.vocabulary = vocabulary
        tokenizer.char_to_idx = {char: idx for idx, char in enumerate(tokenizer.vocabulary)}
        tokenizer.idx_to_char = {idx: char for idx, char in enumerate(tokenizer.vocabulary)}
        tokenizer.unk_token_id = tokenizer.char_to_idx.get('<unk>')
        
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