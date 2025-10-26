"""
Unit tests for the evaluation module.
"""
from pocket_narrator.evaluate import run_evaluation

def test_run_evaluation_mvp_perfect_match():
    """Tests the MVP accuracy with a perfect match on a batch."""
    # Arrange
    predicted_tokens = [[3, 4, 5]]  # Model predicts "sat" then garbage
    target_tokens = [[3, 9, 2]]      # Target is "sat" then something else
    # Text arguments are needed for the function signature, but can be dummy for this test
    predicted_text = ["a b c"]
    target_text = ["d e f"]
    
    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert "mvp_accuracy" in summary
    assert summary["mvp_accuracy"] == 1.0

def test_run_evaluation_mvp_mismatch():
    """Tests the MVP accuracy with a mismatch on a batch."""
    # Arrange
    predicted_tokens = [[4, 4, 5]] # Model predicts a wrong first token
    target_tokens = [[3, 9, 2]]
    predicted_text = ["a b c"]
    target_text = ["d e f"]

    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.0

def test_run_evaluation_mvp_multi_item_batch():
    """Tests accuracy with a batch containing multiple items (one correct, one incorrect)."""
    # Arrange
    predicted_tokens = [[3, 1], [8, 2]] 
    target_tokens = [[3, 9], [7, 1]]
    predicted_text = ["a", "b"]
    target_text = ["c", "d"]
    
    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.5 # 1 out of 2 is correct

def test_run_evaluation_empty_batch():
    """Tests that the evaluation handles an empty batch gracefully."""
    # Arrange
    predicted_tokens = []
    target_tokens = []
    predicted_text = []
    target_text = []

    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.0