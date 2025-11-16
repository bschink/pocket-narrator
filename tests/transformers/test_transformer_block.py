"""
Unit tests for the TransformerBlock component.
"""
import pytest
import torch
from pocket_narrator.models.transformers.transformer_block import TransformerBlock
from pocket_narrator.models.transformers.base_attention import AbstractAttention
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
    output_tensor = transformer_block(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape

def test_dependency_injection_and_mask_propagation(mocker):
    # Arrange
    spy = mocker.spy(MultiHeadSelfAttention, "forward")
    
    attention_module = MultiHeadSelfAttention(d_model=64, n_head=4)
    block = TransformerBlock(d_model=64, attention_module=attention_module, dropout=0.1)
    
    input_tensor = torch.randn(2, 10, 64)
    mask = torch.ones(10, 10)

    # Act
    block(input_tensor, mask=mask)

    # Assert
    # check that the 'forward' method of attention module was called exactly once
    spy.assert_called_once()
    
    # check that the mask passed to the block was the same mask received by the attention module
    passed_mask = spy.call_args.kwargs['mask']
    assert torch.equal(passed_mask, mask)