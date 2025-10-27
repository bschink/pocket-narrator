"""
Unit tests for the tokenizer module, including the SimpleTokenizer class
and the get_tokenizer factory function.
"""
import pytest
from pocket_narrator.tokenizer import SimpleTokenizer, get_tokenizer

# --- Tests for the SimpleTokenizer Class ---

def test_simple_tokenizer_initialization():
    """
    Tests that the tokenizer initializes correctly, creating the necessary mappings
    and identifying the special token IDs.
    """
    # Arrange & Act
    tokenizer = SimpleTokenizer()
    
    # Assert
    assert tokenizer.get_vocab_size() == 27 # Based on the hardcoded vocabulary
    assert tokenizer.unk_token_id == 1      # <unk> is the second item in the list (index 1)
    assert tokenizer.char_to_idx['a'] == 5
    assert tokenizer.idx_to_char[5] == 'a'

def test_encode_simple_string():
    """Tests encoding a simple string containing only known characters."""
    # Arrange
    tokenizer = SimpleTokenizer()
    text = "a bad cat"
    
    # Act
    token_ids = tokenizer.encode(text)
    
    # Assert
    # Manually calculated from the vocabulary: 'a' 5, ' ' 4, 'b' 6, 'd' 8, 'c' 7, 't' 23
    expected_ids = [5, 4, 6, 5, 8, 4, 7, 5, 23]
    assert token_ids == expected_ids

def test_decode_simple_ids():
    """Tests decoding a simple list of token IDs."""
    # Arrange
    tokenizer = SimpleTokenizer()
    token_ids = [5, 4, 6, 5, 8, 4, 7, 5, 23]
    
    # Act
    text = tokenizer.decode(token_ids)
    
    # Assert
    expected_text = "a bad cat"
    assert text == expected_text

def test_encode_with_unknown_characters():
    """
    Tests that any character not in the hardcoded vocabulary is correctly
    mapped to the <unk> token ID.
    """
    # Arrange
    tokenizer = SimpleTokenizer()
    text = "a catz!" # 'z' and '!' are not in the vocabulary
    unk_id = tokenizer.unk_token_id
    
    # Act
    token_ids = tokenizer.encode(text)
    
    # Assert
    # Expected: 'a' 5, ' ' 4, 'c' 7, 'a' 5, 't' 23, 'z' <unk>, '!' <unk>
    expected_ids = [5, 4, 7, 5, 23, unk_id, unk_id]
    assert token_ids == expected_ids

def test_roundtrip_with_unknowns():
    """
    Tests that encoding and then decoding a string with unknown characters
    produces a predictable (though not identical) result.
    """
    # Arrange
    tokenizer = SimpleTokenizer()
    original_text = "a catz!"
    
    # Act
    token_ids = tokenizer.encode(original_text)
    reconstructed_text = tokenizer.decode(token_ids)
    
    # Assert
    # The unknown characters 'z' and '!' should be decoded back to the <unk> string.
    expected_text = "a cat<unk><unk>"
    assert reconstructed_text == expected_text

def test_batch_methods_are_symmetrical():
    """
    Tests that the batch encoding and decoding methods work correctly and
    are symmetrical for known characters.
    """
    # Arrange
    tokenizer = SimpleTokenizer()
    texts = ["a bad cat", "a sad dad"]
    
    # Act
    encoded_batch = tokenizer.encode_batch(texts)
    decoded_batch = tokenizer.decode_batch(encoded_batch)
    
    # Assert
    # Check that the batch encoding produced the correct list of lists
    expected_encoded = [
        [5, 4, 6, 5, 8, 4, 7, 5, 23], # "a bad cat"
        [5, 4, 22, 5, 8, 4, 8, 5, 8]  # "a sad dad"
    ]
    assert encoded_batch == expected_encoded
    
    # Check that the decoded batch matches the original input
    assert decoded_batch == texts

# --- Tests for the get_tokenizer Factory Function ---

def test_get_tokenizer_success():
    """
    Tests that the factory function returns an instance of the correct class
    when a valid tokenizer type is requested.
    """
    # Act
    tokenizer = get_tokenizer(tokenizer_type="simple")
    
    # Assert
    assert isinstance(tokenizer, SimpleTokenizer)

def test_get_tokenizer_failure():
    """
    Tests that the factory function raises a ValueError when an unknown
    tokenizer type is requested. This ensures the function is robust.
    """
    # Act & Assert
    with pytest.raises(ValueError):
        get_tokenizer(tokenizer_type="bpe_is_not_implemented_yet")