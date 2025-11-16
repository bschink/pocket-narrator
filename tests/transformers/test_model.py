"""
Unit tests for the main TransformerModel class.
"""
import pytest
import torch
import os

from pocket_narrator.models import get_model, load_model
from pocket_narrator.models.transformers.model import TransformerModel

# fixtures

@pytest.fixture
def transformer_config():
    """Provides a standard configuration dictionary for the Transformer."""
    return {
        "model_type": "transformer",
        "vocab_size": 50,
        "d_model": 64,
        "n_layers": 2,
        "n_head": 4,
        "max_len": 128,
        "dropout": 0.1,
        "pos_encoding_type": "sinusoidal",
        "attention_type": "multi_head",
        "eos_token_id": 3,
    }

@pytest.fixture
def transformer_model(transformer_config):
    """Provides a standard TransformerModel instance created via the factory."""
    return get_model(**transformer_config)

# tests

def test_model_initialization(transformer_model, transformer_config):
    assert isinstance(transformer_model, TransformerModel)
    assert transformer_model.config["vocab_size"] == transformer_config["vocab_size"]
    assert len(transformer_model.blocks) == transformer_config["n_layers"]
    assert transformer_model.token_embedding.num_embeddings == transformer_config["vocab_size"]
    assert transformer_model.lm_head.out_features == transformer_config["vocab_size"]

def test_forward_pass_returns_correct_shape(transformer_model):
    # Arrange
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, transformer_model.config["vocab_size"], (batch_size, seq_len))
    
    # Act
    logits = transformer_model(input_ids)
    
    # Assert
    assert logits.shape == (batch_size, seq_len, transformer_model.config["vocab_size"])

def test_predict_is_deterministic_in_greedy_mode(transformer_model):
    # Arrange
    prompt = [[10, 20, 30]] # batch with a single prompt
    
    # Act
    prediction1 = transformer_model.predict_sequence_batch(prompt, strategy="greedy", max_length=10)
    prediction2 = transformer_model.predict_sequence_batch(prompt, strategy="greedy", max_length=10)
    
    # Assert
    assert prediction1 == prediction2

def test_predict_respects_max_length(transformer_model):
    # Arrange
    prompt = [[10, 20]]
    max_gen_length = 5
    
    # Act
    prediction = transformer_model.predict_sequence_batch(prompt, max_length=max_gen_length)
    
    # Assert
    assert len(prediction[0]) == max_gen_length

def test_predict_runs_without_gradient_calculation(transformer_model):
    # Arrange
    prompt = [[10, 20, 30]]
    
    # Act
    idx = torch.tensor(prompt)
    logits = transformer_model(idx)
    
    # Assert
    transformer_model.eval()
    with torch.no_grad():
        logits_no_grad = transformer_model(idx)

    assert not logits_no_grad.requires_grad

def test_save_and_load_roundtrip(transformer_model, tmp_path):
    # Arrange
    model_path = tmp_path / "test_transformer.pth"
    
    input_prompt = [[10, 20, 30, 40]]
    
    # Act
    original_model_prediction = transformer_model.predict_sequence_batch(input_prompt, strategy="greedy")
    
    transformer_model.save(model_path)
    assert os.path.exists(model_path)

    loaded_model = load_model(model_path)
    
    loaded_model_prediction = loaded_model.predict_sequence_batch(input_prompt, strategy="greedy")

    # Assert
    assert isinstance(loaded_model, TransformerModel)
    assert loaded_model.config == transformer_model.config
    assert loaded_model_prediction == original_model_prediction