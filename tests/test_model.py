"""
Unit tests for the model module, including the model classes and the
factory and loading functions.
"""
import os
import pytest
from pocket_narrator.model import get_model, load_model, PocketNarratorModelMVP, AbstractLanguageModel

# --- Tests for the Factory and Loading Functions ---

def test_get_model_factory_success():
    """Tests that the get_model factory returns an instance of the correct class."""
    # Act
    model = get_model(model_type="mvp", vocab_size=50)
    
    # Assert
    assert isinstance(model, PocketNarratorModelMVP)
    assert isinstance(model, AbstractLanguageModel)
    assert model.vocab_size == 50

def test_get_model_factory_failure():
    """Tests that the get_model factory raises a ValueError for an unknown type."""
    with pytest.raises(ValueError, match="Unknown model type: 'transformer'"):
        get_model(model_type="transformer", vocab_size=50)

def test_save_and_load_roundtrip(tmp_path):
    """
    Tests that a model can be saved and then loaded back correctly using the
    master load_model function, which should correctly detect the model type.
    """
    # Arrange
    original_model = get_model(model_type="mvp", vocab_size=123)
    # 'tmp_path' is a special pytest fixture that provides a temporary directory
    model_file = tmp_path / "test_model.pth"

    # Act (Save)
    original_model.save(model_file)
    
    # Assert (File exists)
    assert os.path.exists(model_file)

    # Act (Load using the master loader)
    loaded_model = load_model(model_file)
    
    # Assert (Loaded correctly)
    assert isinstance(loaded_model, PocketNarratorModelMVP)
    assert loaded_model.vocab_size == original_model.vocab_size

def test_load_model_file_not_found():
    """Tests that load_model raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_path.pth")

def test_load_model_missing_type_raises_error(tmp_path):
    """Tests that load_model fails if the model file is corrupted (missing model_type)."""
    # Arrange: Create a corrupted file
    bad_model_file = tmp_path / "bad_model.pth"
    with open(bad_model_file, 'w') as f:
        f.write("vocab_size=50") # Missing the 'model_type' line

    # Act & Assert
    with pytest.raises(ValueError, match="is missing 'model_type' config"):
        load_model(bad_model_file)

# --- Tests for the Concrete MVP Model Class ---

def test_mvp_model_predict_sequence_batch():
    """
    Tests the prediction logic of the concrete MVP model class to ensure it
    respects the batch-in, batch-out contract.
    """
    # Arrange
    model = PocketNarratorModelMVP(vocab_size=10)
    input_batch = [[1, 2], [3, 4, 5]] # A batch with two different prompts
    
    # Act
    prediction_batch = model.predict_sequence_batch(input_batch)
    
    # Assert
    # 1. The output batch should have the same number of items as the input batch.
    assert len(prediction_batch) == 2
    
    # 2. Each item in the output batch should be the hardcoded prediction.
    expected_prediction = [22, 19, 23, 4, 17, 5, 23]
    assert prediction_batch[0] == expected_prediction
    assert prediction_batch[1] == expected_prediction