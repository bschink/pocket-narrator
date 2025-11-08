"""
This module serves as the public-facing interface
for creating, loading, and interacting with all model architectures.
"""
import os
import json
from .base_model import AbstractLanguageModel
from .ngram_model import NGramModel

def get_model(model_type: str, vocab_size: int, **kwargs) -> AbstractLanguageModel:
    """
    Factory function to get a model instance.
    """
    print(f"INFO: Getting model of type '{model_type}'...")
    if model_type == "ngram":
        return NGramModel(vocab_size=vocab_size, n=kwargs.get('n'), eos_token_id=kwargs.get('eos_token_id'))
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

def load_model(model_path: str) -> AbstractLanguageModel:
    """
    Loads a model artifact from a file, automatically detecting its type.
    """
    print(f"INFO: Reading model configuration from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # read configuration from the file
    with open(model_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    config = full_config.get("config", {})
            
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError(f"Model file at {model_path} is missing 'model_type' config.")

    # use model_type to get the correct model class
    if model_type == "ngram":
        ModelClass = NGramModel
    else:
        raise ValueError(f"Unknown model type '{model_type}' found in model file.")

    model = ModelClass.load(model_path, config)
    return model