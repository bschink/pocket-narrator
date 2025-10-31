"""
Unit tests for the tokenizer module, including the CharacterTokenizer class
and the get_tokenizer factory function.
"""
import os
import pytest
from pocket_narrator.tokenizer import CharacterTokenizer, get_tokenizer

# --- Tests for the CharacterTokenizer Class ---

def test_character_tokenizer_initialization():
    tokenizer = CharacterTokenizer()
    assert tokenizer.get_vocab_size() == 0
    assert tokenizer.vocabulary == []
    assert not tokenizer.char_to_idx
    assert tokenizer.unk_token_id is None

def test_train_method_builds_vocab_correctly():
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
    tokenizer = CharacterTokenizer()
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.encode("hello")
    with pytest.raises(RuntimeError, match="Tokenizer has not been trained"):
        tokenizer.decode([1, 2, 3])

def test_save_and_load_roundtrip(tmp_path):
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
    # Arrange
    tokenizer = CharacterTokenizer()
    tokenizer.train(["abc"])
    
    # Act
    text_with_unknowns = "ad"  # 'a' is known, 'd' is unknown
    encoded = tokenizer.encode(text_with_unknowns)
    decoded = tokenizer.decode(encoded)
    
    # Assert
    unk_id = tokenizer.unk_token_id
    assert encoded == [tokenizer.char_to_idx['a'], unk_id]
    assert decoded == "a<unk>"


# --- Tests for the get_tokenizer Factory Function ---

def test_get_tokenizer_loads_existing_file(tmp_path):
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
    assert loaded_tokenizer.get_vocab_size() == 4 + 4 # 4 special + h,e,l,o

def test_get_tokenizer_trains_and_saves_if_nonexistent(tmp_path):
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
    assert tokenizer.get_vocab_size() > 4 # ensure it's not blank
    assert os.path.exists(tokenizer_path)

def test_get_tokenizer_raises_error_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        get_tokenizer(tokenizer_type="some_future_tokenizer")

def test_get_tokenizer_raises_error_when_no_path_or_corpus():
    with pytest.raises(ValueError, match="Must provide either a valid tokenizer_path or a train_corpus"):
        get_tokenizer(tokenizer_type="character")