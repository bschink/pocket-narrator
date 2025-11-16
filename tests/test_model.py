"""
Unit tests for the model package.

This file tests the model factory functions and the concrete implementation
of the NGramModel to ensure its logic is correct.
"""
import os
import pytest
import json

from pocket_narrator.models import get_model, load_model
from pocket_narrator.models.base_model import AbstractLanguageModel
from pocket_narrator.models.ngram_model import NGramModel

# --- Tests for the Factory and Loading Functions ---

def test_get_model_factory_for_ngram():
    """Tests that the get_model factory returns an instance of the NGramModel class."""
    # Arrange
    model_config = {"n": 3, "eos_token_id": 99}
    
    # Act
    model = get_model(model_type="ngram", vocab_size=50, **model_config)
    
    # Assert
    assert isinstance(model, NGramModel)
    assert isinstance(model, AbstractLanguageModel)
    assert model.vocab_size == 50
    assert model.n == 3
    assert model.eos_token_id == 99

def test_get_model_factory_failure_for_unknown_type():
    """Tests that the get_model factory raises a ValueError for an unknown type."""
    with pytest.raises(ValueError, match="Unknown model type: 'unknown_model'"):
        get_model(model_type="unknown_model", vocab_size=50)

def test_load_model_file_not_found():
    """Tests that load_model raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_path.json")

# --- Tests for the Concrete NGramModel Class ---

def test_ngram_model_train_builds_counts():
    """
    Tests that the NGramModel's train() method correctly builds the n-gram
    and context counts from a tokenized corpus.
    """
    # Arrange
    # Corpus: the sequence (0, 1) is followed by 2 twice, and by 3 once.
    train_corpus = [[0, 1, 2], [0, 1, 3], [0, 1, 2]]
    model = NGramModel(vocab_size=10, n=3, eos_token_id=9)
    
    # Act
    model.train(train_corpus)
    
    # Assert
    context = (0, 1)
    assert model.ngram_counts[context][2] == 2
    assert model.ngram_counts[context][3] == 1
    assert model.context_counts[context] == 3

def test_ngram_model_predict_chooses_most_likely_token():
    """Tests that prediction for a known context returns the most frequent next token."""
    # Arrange
    train_corpus = [[0, 1, 2], [0, 1, 3], [0, 1, 2]] # (0,1) -> 2 is most likely
    model = NGramModel(vocab_size=10, n=3, eos_token_id=9)
    model.train(train_corpus)
    
    # Act
    prediction = model.predict_sequence_batch(input_tokens_batch=[[0, 1]], max_length=1)
    
    # Assert
    assert prediction[0][0] == 2

def test_ngram_model_predict_stops_at_eos():
    """
    Tests that the generation process correctly stops if the model predicts the
    end-of-sequence token.
    """
    # Arrange
    eos_id = 9
    train_corpus = [[0, 1, 2, eos_id], [5, 1, 2, eos_id]]
    model = NGramModel(vocab_size=10, n=3, eos_token_id=eos_id)
    model.train(train_corpus)
    
    # Act:
    prediction = model.predict_sequence_batch(input_tokens_batch=[[0, 1, 2]], max_length=5)
    
    # Assert
    assert prediction[0] == []

def test_ngram_model_save_and_load_roundtrip(tmp_path):
    """
    Tests that a trained NGramModel can be saved to a JSON file and then loaded back,
    restoring its full state and functionality.
    """
    # Arrange
    eos_id = 9
    train_corpus = [[0, 1, 2, eos_id]]
    original_model = NGramModel(vocab_size=10, n=3, eos_token_id=eos_id)
    original_model.train(train_corpus)
    model_path = tmp_path / "ngram_test.json"

    # Act
    original_model.save(model_path)
    loaded_model = load_model(model_path)
    
    # Assert
    assert isinstance(loaded_model, NGramModel)
    assert loaded_model.n == original_model.n
    assert loaded_model.vocab_size == original_model.vocab_size
    assert loaded_model.eos_token_id == original_model.eos_token_id
    
    original_context = (0, 1)
    loaded_context = (0, 1)
    assert loaded_model.ngram_counts[loaded_context] == original_model.ngram_counts[original_context]
    
    prediction = loaded_model.predict_sequence_batch(input_tokens_batch=[[0, 1]], max_length=1)
    assert prediction[0][0] == 2