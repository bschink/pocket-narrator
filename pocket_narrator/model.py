"""
This module contains the model architectures for the PocketNarrator project.
It defines an abstract base class and provides a factory
function to instantiate specific model implementations.
"""
import os
from abc import ABC, abstractmethod

# --- The Abstract Base Class ---

class AbstractLanguageModel(ABC):
    """
    An abstract base class defining the interface for all models in this project.
    """
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        super().__init__()

    @abstractmethod
    def predict_sequence_batch(self, input_tokens_batch: list[list[int]]) -> list[list[int]]:
        """Predicts a sequence of tokens given a batch of input prompts."""
        pass

    @abstractmethod
    def save(self, model_path: str):
        """Saves the model artifact to a file."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, config: dict):
        """Loads a model artifact. Takes a pre-parsed config dict."""
        pass

# --- The MVP Model ---

class PocketNarratorModelMVP(AbstractLanguageModel):
    """A placeholder MVP model that follows the AbstractLanguageModel contract."""

    def predict_sequence_batch(self, input_tokens_batch: list[list[int]]) -> list[list[int]]:
        print(f"MVP: Model predicting from batch of size {len(input_tokens_batch)}...")
        hardcoded_prediction = [22, 5, 23, 4, 19, 18, 4, 23, 12, 9, 4, 17, 5, 23] # "sat on the mat"
        return [hardcoded_prediction] * len(input_tokens_batch)

    def save(self, model_path: str):
        print(f"MVP: Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # saving config
        with open(model_path, 'w') as f:
            f.write(f"model_type=mvp\n")
            f.write(f"vocab_size={self.vocab_size}\n")
        print("MVP: Model saved successfully.")
    
    @classmethod
    def load(cls, model_path: str, config: dict):
        """
        Loads the MVP model
        """
        print("MVP: Instantiating model from config...")
        vocab_size = int(config['vocab_size'])
        # The 'model_path' argument is unused for the MVP. later call torch.load(model_path).
        return cls(vocab_size=vocab_size)

# --- The Factory Function ---

def get_model(model_type: str, vocab_size: int, **kwargs) -> AbstractLanguageModel:
    """
    Factory function to get a model instance.
    This is the single entry point for creating models in the application.
    """
    print(f"INFO: Getting model of type '{model_type}'...")
    if model_type == "mvp":
        return PocketNarratorModelMVP(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
    
# --- The Master Loading Function ---

def load_model(model_path: str) -> AbstractLanguageModel:
    """
    Loads a model artifact from a file, automatically detecting its type

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        An initialized and loaded model instance.
    """
    print(f"INFO: Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # read the configuration from the file into a dictionary
    config = {}
    with open(model_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                config[key] = val
            
    # determine model type from the file
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError(f"Model file at {model_path} is missing 'model_type' config.")

    # Use the model_type to get the correct model CLASS.
    if model_type == "mvp":
        ModelClass = PocketNarratorModelMVP
    else:
        raise ValueError(f"Unknown model type '{model_type}' found in model file.")

    model = ModelClass.load(model_path, config)
    
    return model