import pytest
import torch
from pocket_narrator.models.transformers.attention import MultiHeadSelfAttention

@pytest.fixture
def attention_module():
    d_model = 64
    n_head = 4
    dropout = 0.1
    return MultiHeadSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)

def test_forward_pass_preserves_shape(attention_module):
    # Arrange
    batch_size = 4
    seq_len = 50
    d_model = 64
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    # Act
    output_tensor = attention_module(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape

def test_causal_masking_works(attention_module):
    # Arrange
    attention_module.eval()
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    # Create a causal mask (upper triangular)
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Create two input tensors. The second one has a change in a future position.
    input_1 = torch.randn(batch_size, seq_len, d_model)
    input_2 = input_1.clone()
    input_2[:, 3, :] = torch.randn(d_model) # Perturb the token at position 3

    # Act
    output_1 = attention_module(input_1, mask=mask)
    output_2 = attention_module(input_2, mask=mask)

    # Assert
    position_to_check = 2
    assert torch.allclose(output_1[:, position_to_check, :], output_2[:, position_to_check, :], atol=1e-6)

    position_after_change = 4
    assert not torch.allclose(output_1[:, position_after_change, :], output_2[:, position_after_change, :])

def test_dropout_is_applied_in_train_mode(attention_module):
    attention_module.train()
    input_tensor = torch.randn(2, 10, 64)

    output_1 = attention_module(input_tensor)
    output_2 = attention_module(input_tensor)
    assert not torch.equal(output_1, output_2)

def test_dropout_is_disabled_in_eval_mode(attention_module):
    attention_module.eval()
    input_tensor = torch.randn(2, 10, 64)

    output_1 = attention_module(input_tensor)
    output_2 = attention_module(input_tensor)
    assert torch.equal(output_1, output_2)