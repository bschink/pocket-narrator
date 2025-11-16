"""
The tokenizers package. This module serves as the public-facing interface
for instantiating or loading tokenizers.
"""
import os
from .base_tokenizer import AbstractTokenizer
from .character_tokenizer import CharacterTokenizer
# from .bpe_tokenizer import BPETokenizer ---> moved downstairs

def get_tokenizer(
    tokenizer_type: str, 
    tokenizer_path: str = None, 
    **kwargs
) -> AbstractTokenizer:
    """
    Factory function to get a tokenizer instance.

    - If tokenizer_path exists, it loads a pre-trained tokenizer from that directory.
    - If it doesn't, it returns a new, blank instance of the specified type,
      configured with the provided kwargs.
    """
    if tokenizer_type == "character":
        TokenizerClass = CharacterTokenizer
    elif tokenizer_type == "bpe":
        from .bpe_tokenizer import BPETokenizer
        TokenizerClass = BPETokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: '{tokenizer_type}'")
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"INFO: Loading existing {TokenizerClass.__name__} from {tokenizer_path}.")
        return TokenizerClass.load(tokenizer_path)
    else:
        print(f"INFO: Creating a new, untrained {TokenizerClass.__name__}.")
        return TokenizerClass(**kwargs)