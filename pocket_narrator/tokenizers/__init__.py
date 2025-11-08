"""
The tokenizers package. This module serves as the public-facing interface
for creating, loading, and managing all tokenizer implementations.
"""
import os
from .base_tokenizer import AbstractTokenizer
from .character_tokenizer import CharacterTokenizer

def get_tokenizer(
    tokenizer_type: str, 
    tokenizer_path: str = None, 
    train_corpus: list[str] = None
) -> AbstractTokenizer:
    """
    Factory function to manage the lifecycle of a tokenizer.

    - If tokenizer_path exists, it loads a pre-trained tokenizer.
    - If it doesn't exist, it trains a new one using train_corpus and saves it.
    """
    if tokenizer_type == "character":
        TokenizerClass = CharacterTokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: '{tokenizer_type}'")

    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"INFO: Loading existing {TokenizerClass.__name__} from {tokenizer_path}.")
        return TokenizerClass.load(tokenizer_path)
    elif train_corpus:
        print(f"INFO: No existing tokenizer found. Training a new {TokenizerClass.__name__}.")
        tokenizer = TokenizerClass()
        tokenizer.train(train_corpus)
        if tokenizer_path:
            tokenizer.save(tokenizer_path)
        return tokenizer
    else:
        raise ValueError(f"Must provide either a valid tokenizer_path or a train_corpus for '{tokenizer_type}' tokenizer.")