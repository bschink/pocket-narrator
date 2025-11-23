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
    output_tensor, present = attention_module(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape
    assert isinstance(present, tuple)
    assert len(present) == 2

def test_causal_masking_works(attention_module):
    # Arrange
    attention_module.eval()
    batch_size = 2
    seq_len = 5
    d_model = 64
    
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    input_1 = torch.randn(batch_size, seq_len, d_model)
    input_2 = input_1.clone()
    input_2[:, 3, :] = torch.randn(d_model)

    # Act
    output_1, _ = attention_module(input_1, mask=mask)
    output_2, _ = attention_module(input_2, mask=mask)

    # Assert
    position_to_check = 2
    assert torch.allclose(output_1[:, position_to_check, :], output_2[:, position_to_check, :], atol=1e-6)

    position_after_change = 4
    assert not torch.allclose(output_1[:, position_after_change, :], output_2[:, position_after_change, :])

def test_dropout_is_applied_in_train_mode(attention_module):
    attention_module.train()
    input_tensor = torch.randn(2, 10, 64)

    output_1, _ = attention_module(input_tensor)
    output_2, _ = attention_module(input_tensor)
    assert not torch.equal(output_1, output_2)

def test_dropout_is_disabled_in_eval_mode(attention_module):
    attention_module.eval()
    input_tensor = torch.randn(2, 10, 64)

    output_1, _ = attention_module(input_tensor)
    output_2, _ = attention_module(input_tensor)
    assert torch.equal(output_1, output_2)

def test_kv_caching_logic(attention_module):
    """
    Tests that passing a 'layer_past' correctly concatenates with new input.
    """
    attention_module.eval()
    batch_size = 1
    d_model = 64
    
    x_prompt = torch.randn(batch_size, 3, d_model)
    _, present_prompt = attention_module(x_prompt)
    
    past_k, past_v = present_prompt
    assert past_k.shape[2] == 3
    
    x_gen = torch.randn(batch_size, 1, d_model)
    _, present_gen = attention_module(x_gen, layer_past=present_prompt)
    
    new_k, new_v = present_gen
    
    assert new_k.shape[2] == 4
    assert new_v.shape[2] == 4