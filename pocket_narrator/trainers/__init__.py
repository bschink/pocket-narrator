"""
The trainers package. This module provides a factory function to get the
correct trainer for a given model type.
"""
from .base_trainer import AbstractTrainer
from .ngram_trainer import NGramTrainer
from .transformer_trainer import TransformerTrainer

def get_trainer(trainer_type: str, **kwargs) -> AbstractTrainer:
    """
    Factory function to get a trainer instance.
    """
    print(f"INFO: Getting trainer of type '{trainer_type}'...")
    if trainer_type == "ngram":
        return NGramTrainer(**kwargs)
    elif trainer_type == "transformer":
        return TransformerTrainer(**kwargs)
    else:
        raise ValueError(f"Unknown trainer type: '{trainer_type}'")