"""
Unit tests for the TransformerBlock component.
"""
import pytest
import torch
from pocket_narrator.models.transformers.transformer_block import TransformerBlock
from pocket_narrator.models.transformers.attention import MultiHeadSelfAttention

@pytest.fixture
def real_attention_module():
    return MultiHeadSelfAttention(d_model=64, n_head=4)

@pytest.fixture
def transformer_block(real_attention_module):
    return TransformerBlock(d_model=64, attention_module=real_attention_module, dropout=0.1)

def test_forward_pass_preserves_shape(transformer_block):
    # Arrange
    input_tensor = torch.randn(4, 50, 64)
    
    # Act
    output_tensor, present = transformer_block(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape
    assert isinstance(present, tuple)

def test_dependency_injection_and_mask_propagation(mocker):
    # Arrange
    mock_attn = mocker.MagicMock(spec=MultiHeadSelfAttention)
    mock_attn.return_value = (torch.randn(2, 10, 64), None)
    
    block = TransformerBlock(d_model=64, attention_module=mock_attn, dropout=0.1)
    
    input_tensor = torch.randn(2, 10, 64)
    mask = torch.ones(10, 10)

    # Act
    block(input_tensor, mask=mask)

    # Assert
    mock_attn.assert_called_once()
    call_kwargs = mock_attn.call_args.kwargs
    
    assert torch.equal(call_kwargs['mask'], mask)
    assert call_kwargs['layer_past'] is None