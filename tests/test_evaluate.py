"""
Unit tests for the evaluation module.
"""
import math
from unittest.mock import patch, MagicMock
from pocket_narrator.evaluate import run_evaluation, calculate_grammar_score

# --- MVP Accuracy Tests ---

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

# --- Perplexity Tests ---

def test_run_evaluation_calculates_perplexity():
    """Tests that passing a val_loss results in correct perplexity."""
    predicted_tokens = []
    target_tokens = []
    predicted_text = []
    target_text = []
    
    val_loss = 2.0
    expected_ppl = math.exp(val_loss)
    
    summary = run_evaluation(
        predicted_tokens, target_tokens, predicted_text, target_text, 
        val_loss=val_loss, check_grammar=False
    )
    
    assert "perplexity" in summary
    assert summary["perplexity"] == expected_ppl

def test_run_evaluation_handles_none_perplexity():
    """Tests that passing None for val_loss returns None for perplexity."""
    summary = run_evaluation([], [], [], [], val_loss=None, check_grammar=False)
    assert summary["perplexity"] is None

# --- BLEU, ROUGE, Diversity Tests ---

def test_run_evaluation_perfect_text_match_metrics():
    """
    If predicted text exactly matches target text, BLEU and ROUGE should be 1.0.
    """
    predicted_tokens = [[1]]
    target_tokens = [[1]]
    
    predicted_text = ["the cat sat"]
    target_text = ["the cat sat"]
    
    summary = run_evaluation(
        predicted_tokens, target_tokens, predicted_text, target_text, check_grammar=False
    )
    
    # BLEU-4 might be 0 because sequence length (3) < 4, but ROUGE-1/2 should be 1.0
    assert summary["rouge_1"] == 1.0
    assert summary["rouge_2"] == 1.0
    assert summary["rouge_l"] == 1.0
    
    # Distinct-1: "the", "cat", "sat" -> 3/3 = 1.0
    assert summary["distinct_1"] == 1.0
    # Repetition rate: 0
    assert summary["repetition_rate"] == 0.0

def test_run_evaluation_text_mismatch_metrics():
    """
    If texts are completely different, overlap scores should be 0.
    """
    predicted_text = ["apple banana"]
    target_text = ["orange grape"]
    
    summary = run_evaluation(
        [[1]], [[1]], predicted_text, target_text, check_grammar=False
    )
    
    assert summary["rouge_1"] == 0.0
    assert summary["bleu_4"] == 0.0

def test_diversity_metrics_repetitive_text():
    """
    Test that highly repetitive text gets high repetition rate and low distinct scores.
    """
    predicted_text = ["test test test test"]
    target_text = ["irrelevant"]
    
    summary = run_evaluation(
        [[1]], [[1]], predicted_text, target_text, check_grammar=False
    )
    
    # 4 tokens total, 1 unique ("test")
    # Repetition Rate = (4 - 1) / 4 = 0.75
    assert summary["repetition_rate"] == 0.75
    
    # Distinct-1: 1 unique / 4 total = 0.25
    assert summary["distinct_1"] == 0.25

# --- Grammar Checker Tests ---

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_success(mock_load_pipeline):
    """
    Test that the function correctly extracts LABEL_1 scores and averages them.
    """
    mock_pipe = MagicMock()
    
    mock_pipe.return_value = [
        [{"label": "LABEL_1", "score": 0.9}, {"label": "LABEL_0", "score": 0.1}],
        [{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_1", "score": 0.1}] 
    ]
    
    mock_load_pipeline.return_value = mock_pipe
    
    texts = ["Good sentence.", "Bad sentence."]
    
    score = calculate_grammar_score(texts, device="cpu")
    
    assert score == 0.5
    mock_pipe.assert_called_once()

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_mps_device_passing(mock_load_pipeline):
    """
    Test that the 'mps' device string is correctly passed down to the loader.
    """
    mock_pipe = MagicMock()
    mock_pipe.return_value = [[{"label": "LABEL_1", "score": 1.0}]]
    mock_load_pipeline.return_value = mock_pipe
    
    calculate_grammar_score(["test"], device="mps")
    
    mock_load_pipeline.assert_called_with("mps")

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_pipeline_failure(mock_load_pipeline):
    """
    Test that the function returns 0.0 gracefully if the pipeline crashes.
    """
    mock_pipe = MagicMock()
    mock_pipe.side_effect = Exception("CUDA error or something")
    mock_load_pipeline.return_value = mock_pipe
    
    score = calculate_grammar_score(["test"], device="cpu")
    
    assert score == 0.0

def test_calculate_grammar_score_empty_input():
    """
    Test that empty input returns 0.0 without trying to load pipeline.
    """
    score = calculate_grammar_score([], device="cpu")
    assert score == 0.0