"""
Unit tests for the TransformerBlock component.
"""
import pytest
import torch
from pocket_narrator.models.transformers.transformer_block import TransformerBlock, FeedForward
from pocket_narrator.models.transformers.attention import MultiHeadSelfAttention

@pytest.fixture
def real_attention_module():
    return MultiHeadSelfAttention(d_model=64, n_head=4)

@pytest.fixture
def transformer_block(real_attention_module):
    return TransformerBlock(d_model=64, attention_module=real_attention_module, dropout=0.1)

@pytest.fixture
def transformer_block_gelu(real_attention_module):
    return TransformerBlock(d_model=64, attention_module=real_attention_module, dropout=0.1, activation_type="gelu")

@pytest.fixture
def transformer_block_swiglu(real_attention_module):
    return TransformerBlock(d_model=64, attention_module=real_attention_module, dropout=0.1, activation_type="swiglu")


# ============ FeedForward Tests ============

class TestFeedForward:
    def test_gelu_forward_preserves_shape(self):
        # Arrange
        ffn = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="gelu")
        input_tensor = torch.randn(4, 50, 64)
        
        # Act
        output_tensor = ffn(input_tensor)
        
        # Assert
        assert output_tensor.shape == input_tensor.shape

    def test_swiglu_forward_preserves_shape(self):
        # Arrange
        ffn = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="swiglu")
        input_tensor = torch.randn(4, 50, 64)
        
        # Act
        output_tensor = ffn(input_tensor)
        
        # Assert
        assert output_tensor.shape == input_tensor.shape

    def test_invalid_activation_type_raises_error(self):
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unknown activation_type"):
            FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="invalid")

    def test_gelu_and_swiglu_have_similar_parameter_counts(self):
        # Arrange
        ffn_gelu = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="gelu")
        ffn_swiglu = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="swiglu")
        
        # Act
        gelu_params = sum(p.numel() for p in ffn_gelu.parameters())
        swiglu_params = sum(p.numel() for p in ffn_swiglu.parameters())
        
        # Assert
        assert abs(gelu_params - swiglu_params) / gelu_params < 0.2

    def test_activation_type_is_case_insensitive(self):
        # Arrange & Act
        ffn_upper = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="GELU")
        ffn_mixed = FeedForward(d_model=64, expansion_factor=4, dropout=0.1, activation_type="SwiGLU")
        input_tensor = torch.randn(2, 10, 64)
        
        # Assert
        assert ffn_upper(input_tensor).shape == input_tensor.shape
        assert ffn_mixed(input_tensor).shape == input_tensor.shape


# ============ TransformerBlock Tests ============

def test_forward_pass_preserves_shape(transformer_block):
    # Arrange
    input_tensor = torch.randn(4, 50, 64)
    
    # Act
    output_tensor, present = transformer_block(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape
    assert isinstance(present, tuple)

def test_forward_pass_gelu_preserves_shape(transformer_block_gelu):
    # Arrange
    input_tensor = torch.randn(4, 50, 64)
    
    # Act
    output_tensor, present = transformer_block_gelu(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape
    assert isinstance(present, tuple)

def test_forward_pass_swiglu_preserves_shape(transformer_block_swiglu):
    # Arrange
    input_tensor = torch.randn(4, 50, 64)
    
    # Act
    output_tensor, present = transformer_block_swiglu(input_tensor)
    
    # Assert
    assert output_tensor.shape == input_tensor.shape
    assert isinstance(present, tuple)

def test_default_activation_is_gelu(real_attention_module):
    # Arrange
    block = TransformerBlock(d_model=64, attention_module=real_attention_module, dropout=0.1)
    
    # Assert
    assert block.ffn.activation_type == "gelu"

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

def test_dependency_injection_with_swiglu(mocker):
    # Arrange
    mock_attn = mocker.MagicMock(spec=MultiHeadSelfAttention)
    mock_attn.return_value = (torch.randn(2, 10, 64), None)
    
    block = TransformerBlock(d_model=64, attention_module=mock_attn, dropout=0.1, activation_type="swiglu")
    
    input_tensor = torch.randn(2, 10, 64)

    # Act
    output, present = block(input_tensor)

    # Assert
    assert output.shape == input_tensor.shape
    assert block.ffn.activation_type == "swiglu"