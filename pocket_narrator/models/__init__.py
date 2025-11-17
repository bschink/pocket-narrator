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
        from .transformers.model import TransformerModel 
        return TransformerModel.from_config(vocab_size=vocab_size, **kwargs)
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
        
        import torch
        save_dict = torch.load(model_path)
        config = save_dict['config']
        state_dict = save_dict['state_dict']
        
        model = get_model(**config)

        model.load_state_dict(state_dict)
        return model
        
    else:
        raise ValueError(f"Unknown model file extension for {model_path}. Must be .json, .model, or .pth")