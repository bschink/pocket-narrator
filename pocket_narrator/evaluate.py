"""
This module contains all functions related to evaluating the performance of the language models.
The public-facing functions are designed to work on batches of data and accept multiple data representations 
to support a variety of metrics.
"""

def _calculate_mvp_accuracy_on_batch(
    predicted_tokens: list[list[int]], 
    target_tokens: list[list[int]]
) -> float:
    """
    Internal helper for the MVP. Calculates simple accuracy over a batch.
    It checks if the first token of each prediction sequence matches the 
    first token of each target sequence.
    """
    correct_predictions = 0
    total_predictions = len(predicted_tokens)

    if total_predictions == 0:
        return 0.0

    for pred_seq, target_seq in zip(predicted_tokens, target_tokens):
        # For the MVP, we only compare the very first predicted token of the sequence
        if pred_seq and target_seq and pred_seq[0] == target_seq[0]:
            correct_predictions += 1
            
    return correct_predictions / total_predictions


def run_evaluation(
    predicted_tokens: list[list[int]],
    target_tokens: list[list[int]],
    predicted_text: list[str],
    target_text: list[str]
) -> dict:
    """
    Master evaluation function that runs all evaluation metrics and returns a summary dictionary.

    Args:
        predicted_tokens: Batch of predicted token sequences (list of lists of ints).
        target_tokens: Batch of target token sequences (list of lists of ints).
        predicted_text: Batch of decoded predicted sentences (list of strings).
        target_text: Batch of decoded target sentences (list of strings).

    Returns:
        A dictionary of all calculated evaluation metrics.
    """
    print("--- Running Full Evaluation ---")
    
    accuracy = _calculate_mvp_accuracy_on_batch(predicted_tokens, target_tokens)
    # Additional metrics can be added here in the future
    
    evaluation_results = {
        "mvp_accuracy": accuracy,
    }
    
    return evaluation_results