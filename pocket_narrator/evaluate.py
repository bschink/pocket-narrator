"""
This module contains all functions related to evaluating the performance of the language models.
The public-facing functions are designed to work on batches of data and accept multiple data representations 
to support a variety of metrics.
"""

import re
import math
from collections import Counter

def _word_tokenize(text: str) -> list[str]:
    """
    Very simple word tokenizer for evaluation:
    - lowercases
    - uses \b\w+\b to pick out 'word-ish' tokens (letters/digits/_)
    """
    return re.findall(r"\b\w+\b", text.lower())


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

def distinct_n(texts: list[str], n: int = 2) -> float:
    """
    Compute Distinct-n over a list of generated texts.
    
    Distinct-n = (# unique n-grams) / (# total n-grams)

    Args:
        texts: list of generated strings
        n: size of n-grams (1, 2, 3, ...)

    Returns:
        float in [0, 1]. Higher = more diverse / less repetitive.
    """
    if n < 1:
        raise ValueError("n must be >= 1 for distinct-n")

    all_ngrams = []

    for text in texts:
        tokens = _word_tokenize(text)
        if len(tokens) < n:
            continue
        # sliding n-gram window
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))

    if not all_ngrams:
        return 0.0

    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)

def repetition_rate(texts: list[str]) -> float:
    """
    Compute a simple repetition rate over a list of generated texts.

    Definition:
        repetition_rate = (total_tokens - total_unique_tokens) / total_tokens

    Where total_tokens and total_unique_tokens are computed across all texts.

    Returns:
        float in [0, 1]. Higher = more repetition.
    """
    total_tokens = 0
    all_tokens = []

    for text in texts:
        tokens = _word_tokenize(text)
        if not tokens:
            continue
        total_tokens += len(tokens)
        all_tokens.extend(tokens)

    if total_tokens == 0:
        return 0.0

    unique_tokens = set(all_tokens)
    total_unique = len(unique_tokens)

    repeated = total_tokens - total_unique
    return repeated / total_tokens


def run_evaluation(
    predicted_tokens: list[list[int]],
    target_tokens: list[list[int]],
    predicted_text: list[str],
    target_text: list[str]
) -> dict:
    """
    Master evaluation function that runs all evaluation metrics and returns a summary dictionary.

    Args:
        predicted_tokens: Batch of predicted token sequences (list[list[int]]).
        target_tokens: Batch of target token sequences (list[list[int]]).
        predicted_text: Batch of decoded predicted sentences (list[str]).
        target_text: Batch of decoded target sentences (list[str]).

    Returns:
        A dictionary of all calculated evaluation metrics.
    """
    print("--- Running Full Evaluation ---")

    
    # Providing all the evaluation metrics in here:
    evaluation_results = {}
    
    # --- 1. MVP Accuracy
    evaluation_results["mvp_accuracy"] = _calculate_mvp_accuracy_on_batch(predicted_tokens, target_tokens)

    # --- 2. Distinct-n for n=1,2,3
    for n in (1, 2, 3):
        evaluation_results[f"distinct_{n}"] = distinct_n(predicted_text, n=n)
    
    # --- 3. Repetition rate over generated text
    evaluation_results["repetition_rate"] = repetition_rate(predicted_text)
    
    return evaluation_results