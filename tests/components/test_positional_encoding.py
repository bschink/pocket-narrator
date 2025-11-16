"""
Unit tests for the positional encoding modules.
"""
import pytest
import torch
from pocket_narrator.models.components.positional_encoding import SinusoidalPositionalEncoding

# --- Fixtures ---

@pytest.fixture
def pos_encoding_module():
    d_model = 64
    max_len = 128
    dropout = 0.1
    return SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

# --- Tests ---

def test_initialization_shape(pos_encoding_module):
    # Arrange
    d_model = pos_encoding_module.pe.shape[2]
    max_len = pos_encoding_module.pe.shape[1]
    
    # Assert
    assert pos_encoding_module.pe.shape == (1, max_len, d_model)

def test_forward_pass_preserves_shape(pos_encoding_module):
    # Arrange
    batch_size = 4
    seq_len = 50
    d_model = pos_encoding_module.pe.shape[2]
    
    input_tensor = torch.zeros(batch_size, seq_len, d_model)
    
    # Act
    output_tensor = pos_encoding_module(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape

def test_values_are_bounded(pos_encoding_module):
    # raw positional encoding values should be in the expected range.
    assert torch.all(pos_encoding_module.pe >= -1.0)
    assert torch.all(pos_encoding_module.pe <= 1.0)

def test_dropout_is_applied_in_train_mode(pos_encoding_module):
    # Arrange
    pos_encoding_module.train()
    input_tensor = torch.zeros(1, 10, pos_encoding_module.pe.shape[2])
    
    # Act
    output_tensor = pos_encoding_module(input_tensor)
    
    # Assert
    raw_encoding = pos_encoding_module.pe[:, :10, :]
    assert not torch.equal(output_tensor, raw_encoding)

def test_dropout_is_disabled_in_eval_mode(pos_encoding_module):
    # Arrange
    pos_encoding_module.eval()
    input_tensor = torch.zeros(1, 10, pos_encoding_module.pe.shape[2])
    
    # Act
    output_tensor = pos_encoding_module(input_tensor)
    
    # Assert
    expected_output = input_tensor + pos_encoding_module.pe[:, :10, :]
    assert torch.equal(output_tensor, expected_output)