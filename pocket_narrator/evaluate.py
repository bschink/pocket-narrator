"""
This module contains all functions related to evaluating the performance of the language models.
The public-facing functions are designed to work on batches of data and accept multiple data representations 
to support a variety of metrics.
"""

import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import torch
# Temporary: disable grammar scoring because HF CoLA model requires torch>=2.6
ENABLE_GRAMMAR_CHECK = False

# Text quality evaluation (optional)
try:
    from pocket_narrator.text_quality import (
        TextQualityConfig,
        evaluate_text_quality,
        _Embedder
    )
    _HAS_TEXT_QUALITY = True
except ImportError:
    _HAS_TEXT_QUALITY = False

# Noun carryover evaluation (optional)
try:
    from pocket_narrator.noun_carryover import (
        noun_carryover_metrics,
        SoftConfig,
        extract_nouns,
        SoftEmbedder
    )
    _HAS_NOUN_CARRYOVER = True
except ImportError:
    _HAS_NOUN_CARRYOVER = False


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

def _calculate_distinct_n_single(text: str, n: int) -> float:
    """
    Calculate distinct-n for a single text.
    
    Definition:
        distinct_n = (# unique n-grams) / (# total n-grams)
    
    Args:
        text: A single text string
        n: size of n-grams (1, 2, 3, ...)
        
    Returns:
        float in [0, 1]. Higher = more diverse.
    """
    tokens = _word_tokenize(text)
    if len(tokens) < n:
        return 0.0
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))
    
    if not ngrams:
        return 0.0
    
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)


def distinct_n(texts: list[str], n: int = 2) -> float:
    """
    Compute Distinct-n with per-text averaging.

    Args:
        texts: list of text strings
        n: size of n-grams (1, 2, 3, ...)

    Returns:
        float in [0, 1]. Higher = more diverse / less repetitive.
    """
    if n < 1:
        raise ValueError("n must be >= 1 for distinct-n")

    if not texts:
        return 0.0
    
    # Internal batching for consistent structure
    batch_size = min(8, len(texts))
    
    total_score = 0.0
    count = 0
    
    # Process in internal batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            score = _calculate_distinct_n_single(text, n)
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0.0

def _calculate_repetition_rate_single(text: str) -> float:
    """
    Calculate repetition rate for a single text.
    
    Definition:
        repetition_rate = (total_tokens - unique_tokens) / total_tokens
    
    Args:
        text: A single text string
        
    Returns:
        float in [0, 1]. Higher = more repetition.
    """
    tokens = _word_tokenize(text)
    if not tokens:
        return 0.0
    
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    
    repeated = total_tokens - unique_tokens
    return repeated / total_tokens


def repetition_rate(texts: list[str]) -> float:
    """
    Calculates the average repetition rate over a list of texts using per-text scoring.
    
    Mirrors the structure of calculate_grammar_score with:
    - Internal batching for processing efficiency
    - Per-text score calculation
    - Averaged final result
    
    Args:
        texts: list of text strings
        
    Returns:
        float in [0, 1]. Higher = more repetition.
        Returns 0.0 if texts is empty.
    """
    if not texts:
        return 0.0
    
    # Internal batching (similar to grammar_score's batch_size=8)
    batch_size = min(8, len(texts))
    
    total_score = 0.0
    count = 0
    
    # Process in internal batches for consistent structure
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            score = _calculate_repetition_rate_single(text)
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0.0


def _count_words_single(text: str) -> int:
    """
    Count the number of words in a single text.
    
    Args:
        text: A single text string
        
    Returns:
        int: Number of words
    """
    tokens = _word_tokenize(text)
    return len(tokens)


def count_words(texts: list[str]) -> float:
    """
    Calculates the average number of words per text using per-text scoring.
    
    Args:
        texts: list of text strings
        
    Returns:
        float: Average number of words across all texts.
        Returns 0.0 if texts is empty.
    """
    if not texts:
        return 0.0
    
    # Internal batching (similar to grammar_score's batch_size=8)
    batch_size = min(8, len(texts))
    
    total_words = 0
    count = 0
    
    # Process in internal batches for consistent structure
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            words = _count_words_single(text)
            total_words += words
            count += 1
    
    return total_words / count if count > 0 else 0.0


def _count_sentences_single(text: str) -> int:
    """
    Count the number of sentences in a single text.
    Uses simple heuristic: count sentence-ending punctuation (. ! ?)
    
    Args:
        text: A single text string
        
    Returns:
        int: Number of sentences (at least 1 if text is non-empty)
    """
    if not text or not text.strip():
        return 0
    
    # Count sentence-ending punctuation
    sentence_endings = text.count('.') + text.count('!') + text.count('?')
    
    # If no sentence endings found, treat as 1 sentence if text exists
    if sentence_endings == 0:
        return 1 if text.strip() else 0
    
    return sentence_endings


def count_sentences(texts: list[str]) -> float:
    """
    Calculates the average number of sentences per text using per-text scoring.
    
    Args:
        texts: list of text strings
        
    Returns:
        float: Average number of sentences across all texts.
        Returns 0.0 if texts is empty.
    """
    if not texts:
        return 0.0
    
    # Internal batching (similar to grammar_score's batch_size=8)
    batch_size = min(8, len(texts))
    
    total_sentences = 0
    count = 0
    
    # Process in internal batches for consistent structure
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            sentences = _count_sentences_single(text)
            total_sentences += sentences
            count += 1
    
    return total_sentences / count if count > 0 else 0.0


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
    check_grammar: bool = True,
    run_llm_judge: bool = False,
    llm_judge_api_key: Optional[str] = None,
    llm_judge_max_stories: Optional[int] = None,
    llm_judge_prompt_template: Optional[str] = None,
    story_beginnings: Optional[list[str]] = None,
    check_text_quality: bool = True,
    text_quality_config: Optional['TextQualityConfig'] = None,
    check_noun_carryover: bool = True,
    noun_carryover_spacy_model: str = "en_core_web_sm",
    noun_carryover_soft_model: str = "all-MiniLM-L6-v2",
    noun_carryover_threshold: float = 0.70
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
        run_llm_judge: Whether to run LLM-as-a-judge evaluation (requires API key).
        llm_judge_api_key: Google API key for Gemini (reads from env if None).
        llm_judge_max_stories: Max stories to evaluate with LLM judge (None = all).
        llm_judge_prompt_template: Custom prompt template for LLM judge.
        story_beginnings: Story prompts given to the model. Required for LLM judge and noun carryover.
                         If None, uses target_text as story beginnings.
        check_text_quality: Whether to run text quality evaluation (coherence/cohesion).
        text_quality_config: Optional TextQualityConfig for customizing text quality evaluation.
        check_noun_carryover: Whether to run noun carryover evaluation (prompt noun retention).
        noun_carryover_spacy_model: spaCy model for noun extraction (default: "en_core_web_sm").
        noun_carryover_soft_model: Sentence-transformers model for embeddings (default: "all-MiniLM-L6-v2").
        noun_carryover_threshold: Threshold for soft_coverage@tau metric (default: 0.70).

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

    # --- 4b. Word count (average words per text)
    evaluation_results["word_count"] = count_words(predicted_text)
    
    # --- 4c. Sentence count (average sentences per text)
    evaluation_results["sentence_count"] = count_sentences(predicted_text)

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

    # --- 7. LLM-as-a-Judge Evaluation (TinyStories style)
    if run_llm_judge and predicted_text:
        # Use provided story_beginnings or fall back to target_text
        beginnings = story_beginnings if story_beginnings is not None else target_text
        llm_judge_results = run_llm_judge_evaluation(
            story_beginnings=beginnings,
            story_completions=predicted_text,
            prompt_template=llm_judge_prompt_template,
            api_key=llm_judge_api_key,
            max_stories=llm_judge_max_stories
        )
        evaluation_results.update(llm_judge_results)
    else:
        # Add placeholder keys with None values when LLM judge is disabled
        evaluation_results["llm_judge_grammar"] = None
        evaluation_results["llm_judge_creativity"] = None
        evaluation_results["llm_judge_consistency"] = None
        evaluation_results["llm_judge_age_groups"] = None
        evaluation_results["llm_judge_num_evaluated"] = None
        evaluation_results["llm_judge_num_failed"] = None

    # --- 8. Text Quality Evaluation (Coherence + Cohesion)
    if check_text_quality and _HAS_TEXT_QUALITY and predicted_text:
        print("--- Computing Text Quality Metrics (Coherence + Cohesion) ---")
        cfg = text_quality_config or TextQualityConfig()
        
        # Create shared embedder for efficiency
        embedder = None
        if cfg.use_sentence_transformers:
            try:
                embedder = _Embedder(cfg.st_model)
            except Exception as e:
                print(f"WARNING: Could not load sentence-transformers embedder: {e}")
        
        # Compute metrics for each story
        coherence_scores = []
        cohesion_scores = []
        text_quality_scores = []
        
        for story in predicted_text:
            try:
                metrics = evaluate_text_quality(story, cfg=cfg, embedder=embedder)
                coherence_scores.append(metrics.get("coherence", float("nan")))
                cohesion_scores.append(metrics.get("cohesion_mean", float("nan")))
                text_quality_scores.append(metrics.get("text_quality", float("nan")))
            except Exception as e:
                print(f"WARNING: Text quality evaluation failed for one story: {e}")
                coherence_scores.append(float("nan"))
                cohesion_scores.append(float("nan"))
                text_quality_scores.append(float("nan"))
        
        # Aggregate (ignoring NaNs)
        def safe_mean(scores):
            valid = [s for s in scores if not math.isnan(s)]
            return sum(valid) / len(valid) if valid else float("nan")
        
        evaluation_results["text_quality_coherence"] = safe_mean(coherence_scores)
        evaluation_results["text_quality_cohesion"] = safe_mean(cohesion_scores)
        evaluation_results["text_quality_score"] = safe_mean(text_quality_scores)
    else:
        evaluation_results["text_quality_coherence"] = None
        evaluation_results["text_quality_cohesion"] = None
        evaluation_results["text_quality_score"] = None

    # --- 9. Noun Carryover Evaluation (Prompt Noun Retention)
    if check_noun_carryover and _HAS_NOUN_CARRYOVER and predicted_text:
        # Use provided story_beginnings or fall back to target_text as prompts
        prompts = story_beginnings if story_beginnings is not None else target_text
        
        if prompts and len(prompts) == len(predicted_text):
            print("--- Computing Noun Carryover Metrics (Hard + Soft) ---")
            
            from pocket_narrator.noun_carryover import SoftConfig
            soft_cfg = SoftConfig(
                model_name=noun_carryover_soft_model,
                threshold=noun_carryover_threshold
            )
            
            # Aggregate metrics across all prompt-story pairs
            total_hard_coverage = 0.0
            total_hard_jaccard = 0.0
            total_hard_precision = 0.0
            total_soft_coverage = 0.0
            total_soft_at_tau = 0.0
            soft_available_count = 0
            
            for prompt, story in zip(prompts, predicted_text):
                try:
                    metrics = noun_carryover_metrics(
                        prompt,
                        story,
                        spacy_model=noun_carryover_spacy_model,
                        soft_cfg=soft_cfg
                    )
                    
                    total_hard_coverage += metrics["hard_coverage"] or 0.0
                    total_hard_jaccard += metrics["hard_jaccard"] or 0.0
                    total_hard_precision += metrics["hard_precision"] or 0.0
                    
                    # Soft metrics might be None if dependencies missing
                    if metrics["soft_coverage"] is not None:
                        total_soft_coverage += metrics["soft_coverage"]
                        soft_available_count += 1
                    
                    tau_key = f"soft_coverage@{soft_cfg.threshold:.2f}"
                    if metrics.get(tau_key) is not None:
                        total_soft_at_tau += metrics[tau_key]
                        
                except Exception as e:
                    print(f"WARNING: Noun carryover failed for one sample: {e}")
                    continue
            
            n = len(prompts)
            evaluation_results["noun_hard_coverage"] = total_hard_coverage / n if n > 0 else 0.0
            evaluation_results["noun_hard_jaccard"] = total_hard_jaccard / n if n > 0 else 0.0
            evaluation_results["noun_hard_precision"] = total_hard_precision / n if n > 0 else 0.0
            evaluation_results["noun_soft_coverage"] = (
                total_soft_coverage / soft_available_count if soft_available_count > 0 else None
            )
            evaluation_results[f"noun_soft_coverage@{noun_carryover_threshold:.2f}"] = (
                total_soft_at_tau / soft_available_count if soft_available_count > 0 else None
            )
        else:
            print("WARNING: Skipping noun carryover - prompts and predictions length mismatch")
            evaluation_results["noun_hard_coverage"] = None
            evaluation_results["noun_hard_jaccard"] = None
            evaluation_results["noun_hard_precision"] = None
            evaluation_results["noun_soft_coverage"] = None
            evaluation_results[f"noun_soft_coverage@{noun_carryover_threshold:.2f}"] = None
    else:
        evaluation_results["noun_hard_coverage"] = None
        evaluation_results["noun_hard_jaccard"] = None
        evaluation_results["noun_hard_precision"] = None
        evaluation_results["noun_soft_coverage"] = None
        evaluation_results[f"noun_soft_coverage@{noun_carryover_threshold:.2f}"] = None

    
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

    # --- 2b. Word count (average words per text)
    evaluation_results["word_count"] = count_words(dataset_text)
    
    # --- 2c. Sentence count (average sentences per text)
    evaluation_results["sentence_count"] = count_sentences(dataset_text)

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


# =============================================================================
# LLM-as-a-Judge Evaluation (TinyStories paper style)
# =============================================================================

@dataclass
class LLMJudgeResult:
    """
    Aggregated results from LLM-as-a-judge evaluation over multiple stories.
    """
    avg_grammar: float
    avg_creativity: float
    avg_consistency: float
    age_group_distribution: dict[str, int]  # Count of each age group
    individual_scores: list  # List of LLMJudgeScores for each story
    num_evaluated: int
    num_failed: int


# Placeholder prompt template - customize this based on TinyStories paper (page 5)
LLM_JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator of children's stories. 
In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
The exercise tests the student's language abilities and creativity. 
The beginning of the story is wrapped in <story_beginning> and the student's completion is wrapped in <story_completion>.

First provide your general assessment about the part written by the student (<story_completion>).
Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the
student manages to complete the sentence which began in <story_beginning> but wasn't finished if that's the case.

Afterwards, grade the student’s completion in terms of grammar, creativity, consistency with the story’s beginning and
whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be,
as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E:
10-12. F: 13-16. Please evaluate the model's completion based on these criteria, providing a score from 1-3 for each:

1. Grammar: How grammatically correct is the completion?
2. Creativity: How creative and imaginative is the completion?
3. Consistency: How logically consistent is the completion with the beginning?
4. Age group: What age group is this story appropriate for?
   (A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12, F: 13-16)

Story beginning (given in the exercise):
<story_beginning>
{story_beginning}
</story_beginning>

Students completion:
<story_completion>
{story_completion}
</story_completion>

Please frame your evaluation in exactly this format:
<grammar><score 1-3></grammar>
<creativity><score 1-3></creativity>
<consistency><score 1-3></consistency>
<age_group><single letter A-F></age_group>
"""


def calculate_llm_judge_scores(
    story_beginnings: list[str],
    story_completions: list[str],
    prompt_template: Optional[str] = None,
    api_key: Optional[str] = None,
    max_stories: Optional[int] = None
) -> LLMJudgeResult:
    """
    Evaluate generated stories using LLM-as-a-judge method.
    
    This implements the evaluation approach from the TinyStories paper (page 5),
    using an LLM to assess Grammar, Creativity, Consistency, and Age Group.
    
    Args:
        story_beginnings: List of prompts/beginnings given to the model.
        story_completions: List of texts generated by the model.
        prompt_template: Custom prompt template for evaluation. 
                        Use {story_beginning} and {story_completion} as placeholders.
                        If None, uses default template.
        api_key: Google API key for Gemini. If None, reads from GOOGLE_API_KEY env var.
        max_stories: Maximum number of stories to evaluate (for cost control).
                    If None, evaluates all stories.
    
    Returns:
        LLMJudgeResult with aggregated scores and individual evaluations.
    """
    from pocket_narrator.gemini_api import (
        GeminiClient, 
        GeminiAPIError, 
        evaluate_stories_batch,
        LLMJudgeScores
    )
    
    if not story_beginnings or not story_completions:
        return LLMJudgeResult(
            avg_grammar=0.0,
            avg_creativity=0.0,
            avg_consistency=0.0,
            age_group_distribution={},
            individual_scores=[],
            num_evaluated=0,
            num_failed=0
        )
    
    # Use default template if none provided
    template = prompt_template or LLM_JUDGE_PROMPT_TEMPLATE
    
    # Limit number of stories if specified
    beginnings_to_evaluate = story_beginnings[:max_stories] if max_stories else story_beginnings
    completions_to_evaluate = story_completions[:max_stories] if max_stories else story_completions
    
    try:
        client = GeminiClient(api_key=api_key)
        scores = evaluate_stories_batch(
            beginnings_to_evaluate, 
            completions_to_evaluate, 
            template, 
            client=client
        )
    except GeminiAPIError as e:
        print(f"ERROR: Failed to initialize LLM judge: {e}")
        return LLMJudgeResult(
            avg_grammar=0.0,
            avg_creativity=0.0,
            avg_consistency=0.0,
            age_group_distribution={},
            individual_scores=[],
            num_evaluated=0,
            num_failed=len(beginnings_to_evaluate)
        )
    
    # Aggregate results
    total_grammar = 0.0
    total_creativity = 0.0
    total_consistency = 0.0
    age_groups: dict[str, int] = {}
    num_failed = 0
    
    for score in scores:
        if score.age_group == "error":
            num_failed += 1
            continue
            
        total_grammar += score.grammar
        total_creativity += score.creativity
        total_consistency += score.consistency
        
        # Normalize age group string for counting
        age_key = score.age_group.lower().strip()
        age_groups[age_key] = age_groups.get(age_key, 0) + 1
    
    num_success = len(scores) - num_failed
    
    return LLMJudgeResult(
        avg_grammar=total_grammar / num_success if num_success > 0 else 0.0,
        avg_creativity=total_creativity / num_success if num_success > 0 else 0.0,
        avg_consistency=total_consistency / num_success if num_success > 0 else 0.0,
        age_group_distribution=age_groups,
        individual_scores=scores,
        num_evaluated=num_success,
        num_failed=num_failed
    )


def run_llm_judge_evaluation(
    story_beginnings: list[str],
    story_completions: list[str],
    prompt_template: Optional[str] = None,
    api_key: Optional[str] = None,
    max_stories: Optional[int] = None
) -> dict:
    """
    Run LLM-as-a-judge evaluation and return results as a dictionary.
    
    This is the main entry point for LLM-based evaluation, designed to be
    called alongside other evaluation metrics.
    
    Args:
        story_beginnings: List of prompts/beginnings given to the model.
        story_completions: List of texts generated by the model.
        prompt_template: Custom prompt template (uses default if None).
        api_key: Google API key (reads from env if None).
        max_stories: Max stories to evaluate (None = all).
    
    Returns:
        Dictionary with evaluation results:
        - llm_judge_grammar: Average grammar score (1-3)
        - llm_judge_creativity: Average creativity score (1-3)
        - llm_judge_consistency: Average consistency score (1-3)
        - llm_judge_age_groups: Distribution of age groups (A-F)
        - llm_judge_num_evaluated: Number of successfully evaluated stories
        - llm_judge_num_failed: Number of failed evaluations
    """
    print("--- Running LLM-as-Judge Evaluation ---")
    
    result = calculate_llm_judge_scores(
        story_beginnings=story_beginnings,
        story_completions=story_completions,
        prompt_template=prompt_template,
        api_key=api_key,
        max_stories=max_stories
    )
    
    return {
        "llm_judge_grammar": result.avg_grammar,
        "llm_judge_creativity": result.avg_creativity,
        "llm_judge_consistency": result.avg_consistency,
        "llm_judge_age_groups": result.age_group_distribution,
        "llm_judge_num_evaluated": result.num_evaluated,
        "llm_judge_num_failed": result.num_failed,
    }