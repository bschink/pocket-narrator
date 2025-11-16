"""
This module serves as the public-facing interface
for creating, loading, and interacting with all model architectures.
"""
import os
import json
# import torch (moved downstairs)

from .base_model import AbstractLanguageModel
from .ngram_model import NGramModel
# from .transformers.model import TransformerModel 
# from .components.positional_encoding import SinusoidalPositionalEncoding
# from .transformers.attention import MultiHeadSelfAttention
# from .transformers.transformer_block import TransformerBlock

def get_model(model_type: str, vocab_size: int, **kwargs) -> AbstractLanguageModel:
    """
    Factory function to get a model instance.
    """
    print(f"INFO: Getting model of type '{model_type}'...")
    if model_type == "ngram":
        return NGramModel(
            vocab_size=vocab_size, 
            n=kwargs.get('n'), 
            eos_token_id=kwargs.get('eos_token_id')
        )
    elif model_type == "transformer":
        import torch
        from .transformers.model import TransformerModel 
        from .components.positional_encoding import SinusoidalPositionalEncoding
        from .transformers.attention import MultiHeadSelfAttention
        from .transformers.transformer_block import TransformerBlock
        d_model = kwargs.get('d_model', 256)
        max_len = kwargs.get('max_len', 128)
        dropout = kwargs.get('dropout', 0.1)
        n_head = kwargs.get('n_head', 4)
        n_layers = kwargs.get('n_layers', 4)
        
        # positional encoding module
        pos_encoding_type = kwargs.get("pos_encoding_type", "sinusoidal")
        if pos_encoding_type in ["sinusoidal", "SinusoidalPositionalEncoding"]:
            pos_encoding_module = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        # attention module
        attention_type = kwargs.get("attention_type", "multi_head")
        if attention_type in ["multi_head", "MultiHeadSelfAttention"]:
            attention_module = MultiHeadSelfAttention(d_model, n_head, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
            
        block_template = TransformerBlock(d_model, attention_module, dropout)
        
        return TransformerModel(
            vocab_size=vocab_size,
            n_layers=n_layers,
            pos_encoding_module=pos_encoding_module,
            transformer_block_template=block_template
        )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

def load_model(model_path: str) -> AbstractLanguageModel:
    """
    Loads a model artifact from a file, automatically detecting its format
    (JSON for n-gram, PyTorch .pth for neural models).
    """
    model_path = str(model_path)
    print(f"INFO: Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # detect file type
    if model_path.endswith(".json") or model_path.endswith(".model"):
        # ngram model (supports both .json and .model extensions)
        print("INFO: Detected JSON/model file. Using n-gram loading logic.")
        with open(model_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        config = saved_data.get("config", {})
        model_type = config.get("model_type")

        if model_type == "ngram":
            ModelClass = NGramModel
        else:
            raise ValueError(f"Unknown model type '{model_type}' in JSON file.")
        
        model = ModelClass.load(model_path, config)
        return model

    elif model_path.endswith(".pth"):
        # neural model
        print("INFO: Detected PyTorch .pth model file. Using neural model loading logic.")
        
        save_dict = torch.load(model_path)
        config = save_dict['config']
        state_dict = save_dict['state_dict']
        
        model_type = config.pop('model_type')
        vocab_size = config.pop('vocab_size')
        
        model = get_model(
            model_type=model_type,
            vocab_size=vocab_size,
            **config
        )
        
        model.load_state_dict(state_dict)
        
        return model
        
    else:
        raise ValueError(f"Unknown model file extension for {model_path}. Must be .json, .model, or .pth")