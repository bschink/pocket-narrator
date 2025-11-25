"""
This module contains all functions related to evaluating the performance of the language models.
The public-facing functions are designed to work on batches of data and accept multiple data representations 
to support a variety of metrics.
"""

import re
import math
from collections import Counter
import torch
# Temporary: disable grammar scoring because HF CoLA model requires torch>=2.6
ENABLE_GRAMMAR_CHECK = False


try:
    from transformers import pipeline
    _GRAMMAR_PIPELINE = None
except ImportError:
    _GRAMMAR_PIPELINE = None

def _word_tokenize(text: str) -> list[str]:
    """
    Very simple word tokenizer for evaluation:
    - lowercases
    - uses \b\w+\b to pick out 'word-ish' tokens (letters/digits/_)
    """
    return re.findall(r"\b\w+\b", text.lower())

def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Helper to generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


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


def calculate_perplexity(loss_value: float) -> float:
    """
    Calculates perplexity from the average cross-entropy loss.
    PPL = exp(Loss)
    """
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float('inf')
    
def calculate_bleu(candidate_text: str, reference_text: str, max_n: int = 4) -> float:
    """
    Calculates a simplified BLEU score (BLEU-1 to BLEU-4).
    
    Args:
        candidate_text: The generated string.
        reference_text: The ground truth string.
        max_n: Maximum n-gram order to check (usually 4).
    """
    cand_tokens = _word_tokenize(candidate_text)
    ref_tokens = _word_tokenize(reference_text)
    
    if not cand_tokens:
        return 0.0

    # brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0.0

    # n-gram precision (geometric mean)
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _get_ngrams(cand_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        clipped_counts = {
            gram: min(count, ref_ngrams.get(gram, 0)) 
            for gram, count in cand_ngrams.items()
        }
        
        numerator = sum(clipped_counts.values())
        denominator = max(1, sum(cand_ngrams.values())) # avoid div by 0
        
        precisions.append(numerator / denominator)

    if any(p == 0 for p in precisions):
        return 0.0

    # geometric mean: exp(sum(log(p)) / N)
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / max_n)

    return bp * geo_mean

def _lcs_length(x: list[str], y: list[str]) -> int:
    """
    Computes Longest Common Subsequence length using dynamic programming.
    """
    m, n = len(x), len(y)
    # using 2 rows instead of full matrix to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = list(curr)
        
    return curr[n]

def calculate_rouge_l(candidate_text: str, reference_text: str) -> float:
    """
    Calculates ROUGE-L (Recall based on Longest Common Subsequence).
    """
    cand_tokens = _word_tokenize(candidate_text)
    ref_tokens = _word_tokenize(reference_text)
    
    if not ref_tokens:
        return 0.0
        
    lcs_len = _lcs_length(cand_tokens, ref_tokens)
    
    return lcs_len / len(ref_tokens)

def calculate_rouge_n(candidate_text: str, reference_text: str, n: int = 1) -> float:
    """
    Calculates ROUGE-N (N-gram overlap recall).
    """
    cand_tokens = _word_tokenize(candidate_text)
    ref_tokens = _word_tokenize(reference_text)
    
    if not ref_tokens:
        return 0.0
        
    cand_ngrams = _get_ngrams(cand_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)
    
    # intersection count
    matches = 0
    for gram, count in ref_ngrams.items():
        matches += min(count, cand_ngrams.get(gram, 0))
        
    total_ref_ngrams = sum(ref_ngrams.values())
    return matches / total_ref_ngrams if total_ref_ngrams > 0 else 0.0


def _load_grammar_pipeline(device_str: str):
    """
    Lazy loader for the DistilBERT CoLA model.
    """
    global _GRAMMAR_PIPELINE
    
    if _GRAMMAR_PIPELINE is not None:
        return _GRAMMAR_PIPELINE

    print(f"INFO: Loading DistilBERT-CoLA for grammar evaluation on {device_str}...")
    
    try:
        curr_device = torch.device(device_str)
        
        _GRAMMAR_PIPELINE = pipeline(
            "text-classification", 
            model="textattack/distilbert-base-uncased-CoLA",
            device=curr_device, 
            top_k=None # return scores for both labels
        )
    except Exception as e:
        print(f"ERROR: Could not load grammar pipeline: {e}")
        return None
        
    return _GRAMMAR_PIPELINE

def calculate_grammar_score(texts: list[str], device: str = "cpu") -> float:
    """
    Calculates the average 'Linguistic Acceptability' score using DistilBERT-CoLA.
    Returns a float between 0.0 (Unacceptable) and 1.0 (Grammatically Correct).
    """
    try:
        import transformers
    except ImportError:
        print("WARNING: 'transformers' library not found. Skipping grammar check.")
        return 0.0

    pipe = _load_grammar_pipeline(device)
    
    if pipe is None or not texts:
        return 0.0

    try:
        # Use smaller batch size to avoid memory issues
        batch_size = min(8, len(texts))
        results = pipe(texts, batch_size=batch_size, truncation=True, max_length=512)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"WARNING: GPU out of memory. Retrying on CPU...")
            global _GRAMMAR_PIPELINE
            _GRAMMAR_PIPELINE = None  # reset pipeline
            pipe = _load_grammar_pipeline("cpu")
            if pipe is None:
                return 0.0
            try:
                batch_size = min(8, len(texts))
                results = pipe(texts, batch_size=batch_size, truncation=True, max_length=512)
            except Exception as e2:
                print(f"WARNING: Grammar evaluation failed on CPU: {e2}")
                return 0.0
        else:
            print(f"WARNING: Grammar evaluation failed: {e}")
            return 0.0
    except Exception as e:
        print(f"WARNING: Grammar evaluation failed: {e}")
        return 0.0

    total_score = 0.0
    count = 0
    
    for res in results:
        # res is list of dicts: [{'label': 'LABEL_1', 'score': 0.9}, ...]
        # CoLA: LABEL_1 = Acceptable, LABEL_0 = Unacceptable
        
        score = 0.0
        for item in res:
            if item['label'] == 'LABEL_1':
                score = item['score']
                break
        
        total_score += score
        count += 1

    return total_score / count if count > 0 else 0.0


def run_evaluation(
    predicted_tokens: list[list[int]],
    target_tokens: list[list[int]],
    predicted_text: list[str],
    target_text: list[str],
    val_loss: float = None,
    check_grammar: bool = True
) -> dict:
    """
    Master evaluation function that runs all evaluation metrics and returns a summary dictionary.

    Args:
        predicted_tokens: Batch of predicted token sequences (list[list[int]]).
        target_tokens: Batch of target token sequences (list[list[int]]).
        predicted_text: Batch of decoded predicted sentences (list[str]).
        target_text: Batch of decoded target sentences (list[str]).
        val_loss: The Cross Entropy Loss on the validation set.
        check_grammar: Whether to run the grammar checker.

    Returns:
        A dictionary of all calculated evaluation metrics.
    """
    print("--- Running Full Evaluation ---")
    # Providing all the evaluation metrics in here:
    evaluation_results = {}

    # --- 1. Perplexity
    if val_loss is not None:
        evaluation_results["perplexity"] = calculate_perplexity(val_loss)
    else:
        evaluation_results["perplexity"] = None
    
    # --- 2. MVP Accuracy
    evaluation_results["mvp_accuracy"] = _calculate_mvp_accuracy_on_batch(predicted_tokens, target_tokens)

    # --- 3. Distinct-n for n=1,2,3
    for n in (1, 2, 3):
        evaluation_results[f"distinct_{n}"] = distinct_n(predicted_text, n=n)
    
    # --- 4. Repetition rate over generated text
    evaluation_results["repetition_rate"] = repetition_rate(predicted_text)

    # --- 5. N-gram Overlap Metrics (BLEU & ROUGE)
    total_bleu = 0.0
    total_rouge_1 = 0.0
    total_rouge_2 = 0.0
    total_rouge_l = 0.0
    batch_size = len(predicted_text)
    
    if batch_size > 0:
        for pred, ref in zip(predicted_text, target_text):
            total_bleu += calculate_bleu(pred, ref)
            total_rouge_1 += calculate_rouge_n(pred, ref, n=1)
            total_rouge_2 += calculate_rouge_n(pred, ref, n=2)
            total_rouge_l += calculate_rouge_l(pred, ref)
            
        evaluation_results["bleu_4"] = total_bleu / batch_size
        evaluation_results["rouge_1"] = total_rouge_1 / batch_size
        evaluation_results["rouge_2"] = total_rouge_2 / batch_size
        evaluation_results["rouge_l"] = total_rouge_l / batch_size

    # --- 6. Grammar Score (CoLA)
    if check_grammar and ENABLE_GRAMMAR_CHECK and predicted_text:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        evaluation_results["grammar_score"] = calculate_grammar_score(predicted_text, device=device)
    else:
        evaluation_results["grammar_score"] = None

    
    return evaluation_results

def run_dataset_evaluation(
    dataset_text: list[str],
    check_grammar: bool = True
) -> dict:
    """
    Master evaluation function that runs all applicable evaluation metrics on the dataset.

    Args:
        dataset_text: List of sentences in the dataset (list[str]).
        check_grammar: Whether to run the grammar checker.

    Returns:
        A dictionary of all calculated evaluation metrics.
    """
    print("--- Running Dataset Evaluation ---")
    evaluation_results = {}

    # --- 1. Distinct-n for n=1,2,3
    for n in (1, 2, 3):
        evaluation_results[f"distinct_{n}"] = distinct_n(dataset_text, n=n)
    
    # --- 2. Repetition rate over generated text
    evaluation_results["repetition_rate"] = repetition_rate(dataset_text)

    # --- 3. Grammar Score (CoLA)
    if check_grammar and ENABLE_GRAMMAR_CHECK and dataset_text:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        evaluation_results["grammar_score"] = calculate_grammar_score(dataset_text, device=device)
    else:
        evaluation_results["grammar_score"] = None

    
    return evaluation_results