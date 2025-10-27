"""
This module contains tokenizer implementations and a factory function
to select the correct tokenizer for the project.
"""

class SimpleTokenizer:
    """A simple character-level tokenizer for the MVP."""

    def __init__(self):
        """Initializes the tokenizer with a hardcoded vocabulary."""
        # for the mvp small, fixed vocabulary. in the future this will be learned from the training data
        self.vocabulary = ['<pad>', '<unk>', '<bos>', '<eos>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y']
        
        # create mappings for char<->id
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}

        # special token IDs
        self.unk_token_id = self.char_to_idx['<unk>']

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        """
        Converts a string of text into a list of token IDs.
        Characters not in the vocabulary will be mapped to the <unk> token.
        """
        token_ids = []
        for char in text:
            # look up char in map; if not found, use the <unk> token ID
            token_ids.append(self.char_to_idx.get(char, self.unk_token_id))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Converts a list of token IDs back into a string of text.
        """
        return "".join([self.idx_to_char.get(idx, '') for idx in token_ids])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Applies encoding to a batch (list) of texts."""
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Applies decoding to a batch (list) of token lists."""
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