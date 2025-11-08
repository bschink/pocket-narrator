"""
Unit tests for the tokenizers package.

This file tests both the concrete CharacterTokenizer implementation and the 
get_tokenizer factory function that manages its lifecycle.
"""
import os
import pytest

from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.tokenizers.base_tokenizer import AbstractTokenizer
from pocket_narrator.tokenizers.character_tokenizer import CharacterTokenizer

# --- Tests for the CharacterTokenizer Class ---

def test_character_tokenizer_initialization():
    """Tests that a new CharacterTokenizer instance is blank."""
    tokenizer = CharacterTokenizer()
    assert tokenizer.get_vocab_size() == 0
    assert tokenizer.vocabulary == []
    assert not tokenizer.char_to_idx
    assert tokenizer.unk_token_id is None

def test_train_method_builds_vocab_correctly():
    """
    Tests that the train() method correctly builds the vocabulary from a corpus,
    including special tokens and sorted unique characters.
    """
    # Arrange
    corpus = ["hello", "world"]
    tokenizer = CharacterTokenizer()
    
    # Act
    tokenizer.train(corpus)
    
    # Assert
    # Vocab = 4 special tokens + sorted unique chars from "helloworld" ('d', 'e', 'h', 'l', 'o', 'r', 'w')
    unique_chars = sorted(list(set("".join(corpus))))
    expected_vocab = tokenizer.special_tokens + unique_chars
    
    assert tokenizer.vocabulary == expected_vocab
    assert tokenizer.get_vocab_size() == 4 + 7
    assert tokenizer.char_to_idx['h'] == 6
    assert tokenizer.unk_token_id == 1

def test_untrained_tokenizer_raises_runtime_error():
    """Tests that calling encode() or decode() on an untrained tokenizer fails."""
    tokenizer = CharacterTokenizer()
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.encode("hello")
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.decode([1, 2, 3])

def test_save_and_load_roundtrip(tmp_path):
    """
    Tests that a trained tokenizer can be saved to a file and then loaded back
    to a perfectly identical and functional state.
    """
    # Arrange
    corpus = ["a simple test!"]
    original_tokenizer = CharacterTokenizer()
    original_tokenizer.train(corpus)
    save_path = tmp_path / "vocab.json"

    # Act
    original_tokenizer.save(save_path)
    loaded_tokenizer = CharacterTokenizer.load(save_path)
    
    # Assert
    assert loaded_tokenizer.vocabulary == original_tokenizer.vocabulary
    assert loaded_tokenizer.char_to_idx == original_tokenizer.char_to_idx
    assert loaded_tokenizer.unk_token_id == original_tokenizer.unk_token_id
    
    text = "a test"
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(text)) == text

def test_encode_decode_after_training_with_unknowns():
    """Tests the full encode/decode cycle after training, including unknown characters."""
    # Arrange
    tokenizer = CharacterTokenizer()
    tokenizer.train(["abc"])
    
    # Act
    text_with_unknowns = "ad"
    encoded = tokenizer.encode(text_with_unknowns)
    decoded = tokenizer.decode(encoded)
    
    # Assert
    unk_id = tokenizer.unk_token_id
    assert encoded == [tokenizer.char_to_idx['a'], unk_id]
    assert decoded == "a<unk>"


# --- Tests for the get_tokenizer Factory Function ---

def test_get_tokenizer_loads_existing_file(tmp_path):
    """
    Tests that the factory function correctly loads a pre-existing tokenizer file
    instead of training a new one.
    """
    # Arrange
    corpus1 = ["hello"]
    corpus2 = ["world"]
    tokenizer_path = tmp_path / "vocab.json"
    
    initial_tokenizer = CharacterTokenizer()
    initial_tokenizer.train(corpus1)
    initial_tokenizer.save(tokenizer_path)

    # Act
    loaded_tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_path,
        train_corpus=corpus2
    )

    # Assert
    assert isinstance(loaded_tokenizer, AbstractTokenizer)
    assert loaded_tokenizer.get_vocab_size() == 4 + 4

def test_get_tokenizer_trains_and_saves_if_nonexistent(tmp_path):
    """
    Tests that the factory function trains a new tokenizer if the path is empty
    and a corpus is provided, then saves it to the specified path.
    """
    # Arrange
    corpus = ["new tokenizer"]
    tokenizer_path = tmp_path / "vocab.json"
    
    # Act
    tokenizer = get_tokenizer(
        tokenizer_type="character",
        tokenizer_path=tokenizer_path,
        train_corpus=corpus
    )
    
    # Assert
    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.get_vocab_size() > 4
    assert os.path.exists(tokenizer_path)

def test_get_tokenizer_raises_error_for_unknown_type():
    """Tests that the factory fails gracefully for an invalid tokenizer type."""
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        get_tokenizer(tokenizer_type="some_future_tokenizer")

def test_get_tokenizer_raises_error_when_no_path_or_corpus():
    """
    Tests the critical failure case where a character tokenizer is requested
    without a file to load or a corpus to train on.
    """
    with pytest.raises(ValueError, match="Must provide either a valid tokenizer_path or a train_corpus"):
        get_tokenizer(tokenizer_type="character")