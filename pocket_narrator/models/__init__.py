"""
This module serves as the public-facing interface
for creating, loading, and interacting with all model architectures.
"""
import os
import json
import pickle
# import torch (moved downstairs)

from .base_model import AbstractLanguageModel
from .ngram_model import NGramModel
# from .transformers.model import TransformerModel

def get_model(model_type: str, vocab_size: int, **kwargs) -> AbstractLanguageModel:
    """
    Factory function to get a model instance.

    Supported model types:
      - "ngram"
      - "transformer"
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
    Supports .json, .model (tries PyTorch first, then JSON), and .pth extensions.
    """
    model_path = str(model_path)
    print(f"INFO: Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # detect file type
    if model_path.endswith(".pth"):
        # PyTorch neural model (.pth)
        print("INFO: Detected PyTorch .pth model file. Using neural model loading logic.")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"INFO: Loading model onto {device}...")
        
        save_dict = torch.load(model_path, map_location=device)
        config = save_dict['config']
        state_dict = save_dict['state_dict']
        
        model = get_model(**config)
        model.to(device)
        model.load_state_dict(state_dict)
        return model
    
    elif model_path.endswith(".model"):
        # .model extension: try PyTorch first, then JSON fallback
        import torch
        try:
            print("INFO: Detected .model file. Attempting PyTorch load first...")
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"INFO: Loading model onto {device}...")

            save_dict = torch.load(model_path, map_location=device)
            config = save_dict.get('config', {})
            state_dict = save_dict.get('state_dict')
            
            if config and state_dict:
                # Successfully loaded as PyTorch
                print("INFO: Loaded as PyTorch model (neural/transformer).")
                model = get_model(**config)
                model.to(device)
                model.load_state_dict(state_dict)
                return model
        except (FileNotFoundError, KeyError, pickle.UnpicklingError, RuntimeError, EOFError) as e:
            # PyTorch load failed, try JSON
            print(f"INFO: PyTorch load failed: {e}. Attempting JSON load (n-gram)...")
            pass
        
        # Fallback to JSON (n-gram)
        print("INFO: PyTorch load failed. Attempting JSON load (n-gram)...")
        try:
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
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to load .model file as PyTorch or JSON: {e}")
    
    elif model_path.endswith(".json"):
        # JSON n-gram model
        print("INFO: Detected JSON file. Using n-gram loading logic.")
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
        
    else:
        raise ValueError(f"Unknown model file extension for {model_path}. Must be .json, .model, or .pth")