"""
Unit tests for the trainers package.

This file tests the get_trainer factory function and the concrete logic
of the TransformerTrainer class.
"""
import pytest
import torch

from pocket_narrator.trainers import get_trainer
from pocket_narrator.trainers.base_trainer import AbstractTrainer
from pocket_narrator.trainers.ngram_trainer import NGramTrainer
from pocket_narrator.trainers.transformer_trainer import TransformerTrainer
from pocket_narrator.models import get_model
from pocket_narrator.tokenizers import CharacterTokenizer

# --- Tests for the get_trainer Factory Function ---

def test_get_trainer_factory_for_ngram():
    trainer = get_trainer(trainer_type="ngram")
    assert isinstance(trainer, NGramTrainer)
    assert isinstance(trainer, AbstractTrainer)

def test_get_trainer_factory_for_transformer():
    # Arrange
    trainer_config = {"learning_rate": 1e-5, "epochs": 2}
    
    # Act
    trainer = get_trainer(trainer_type="transformer", **trainer_config)
    
    # Assert
    assert isinstance(trainer, TransformerTrainer)
    assert isinstance(trainer, AbstractTrainer)
    assert trainer.learning_rate == 1e-5
    assert trainer.epochs == 2

def test_get_trainer_factory_failure_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown trainer type: 'unknown_type'"):
        get_trainer(trainer_type="unknown_type")

# --- Tests for the TransformerTrainer's Logic ---

@pytest.fixture
def simple_trainer():
    return TransformerTrainer(batch_size=2)

def test_transformer_trainer_prepare_batch(simple_trainer):
    """
    Tests the critical _prepare_batch_for_lm helper method to ensure it
    correctly creates and pads input (x) and target (y) tensors.
    """
    # Arrange
    batch_tokens = [
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10]
    ]
    max_len = 5

    # Act
    x, y = simple_trainer._prepare_batch_for_lm(batch_tokens, max_len)

    # Assert
    assert x.shape == (2, max_len)
    assert y.shape == (2, max_len)
    
    device = x.device
    expected_x_0 = torch.tensor([1, 2, 3, 0, 0], dtype=torch.long, device=device)
    expected_y_0 = torch.tensor([2, 3, 4, 0, 0], dtype=torch.long, device=device)
    assert torch.equal(x[0], expected_x_0)
    assert torch.equal(y[0], expected_y_0)

    expected_x_1 = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long, device=device)
    expected_y_1 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long, device=device)
    assert torch.equal(x[1], expected_x_1)
    assert torch.equal(y[1], expected_y_1)

def test_transformer_trainer_train_method_updates_weights():
    """
    This is an integration test verifying that the model's weights actually change
    """
    # Arrange
    train_data = ["abcde", "fghij"]
    special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    
    tokenizer = CharacterTokenizer(special_tokens=special_tokens)
    tokenizer.train(train_data)
    
    model_config = {
        "d_model": 16, "n_layers": 1, "n_head": 2, "max_len": 10, "dropout": 0.0,
        "pos_encoding_type": "sinusoidal", "attention_type": "multi_head",
        "eos_token_id": special_tokens['<eos>']
    }
    model = get_model(model_type="transformer", vocab_size=tokenizer.get_vocab_size(), **model_config)
    
    initial_weights = model.lm_head.weight.clone().detach()
    
    trainer = TransformerTrainer(epochs=1, batch_size=2)
    
    # Act
    trained_model = trainer.train(model=model, tokenizer=tokenizer, train_data=train_data)
    
    # Assert
    final_weights = trained_model.lm_head.weight
    assert not torch.equal(initial_weights, final_weights)