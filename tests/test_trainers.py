"""
Unit tests for the trainers package.

This file primarily tests the get_trainer factory function to ensure it
correctly maps trainer types to their corresponding classes.
"""
import pytest

from pocket_narrator.trainers import get_trainer
from pocket_narrator.trainers.base_trainer import AbstractTrainer
from pocket_narrator.trainers.ngram_trainer import NGramTrainer

def test_get_trainer_factory_for_ngram():
    """
    Tests that the get_trainer factory returns an instance of the NGramTrainer class.
    """
    # Act
    trainer = get_trainer(trainer_type="ngram")
    
    # Assert
    assert isinstance(trainer, NGramTrainer)
    assert isinstance(trainer, AbstractTrainer)

def test_get_trainer_factory_failure_for_unknown_type():
    """
    Tests that the get_trainer factory raises a ValueError when an unknown
    trainer type is requested, ensuring robust error handling.
    """
    # Act & Assert
    with pytest.raises(ValueError, match="Unknown trainer type: 'transformer'"):
        get_trainer(trainer_type="transformer")