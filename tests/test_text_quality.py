"""
Unit tests for the text_quality module.

Tests cover:
- Sentence splitting
- Entity extraction (with and without spaCy)
- Coherence computation (Jaccard similarity)
- Cohesion computation (semantic similarity)
- Final quality score computation
- Edge cases and error handling
"""
import math
import pytest
from unittest.mock import patch, MagicMock

from pocket_narrator.text_quality import (
    TextQualityConfig,
    split_sentences,
    _fallback_entities,
    extract_entities,
    jaccard,
    compute_coherence_entity_overlap,
    compute_cohesion_semantic,
    compute_final_quality,
    evaluate_text_quality,
    _Embedder,
)


# =============================================================================
# Sentence Splitting Tests
# =============================================================================

def test_split_sentences_basic():
    """Test basic sentence splitting."""
    cfg = TextQualityConfig()
    text = "Once upon a time. There was a girl. She played."
    sents = split_sentences(text, cfg)
    assert len(sents) == 3
    assert sents[0] == "Once upon a time."
    assert sents[1] == "There was a girl."
    assert sents[2] == "She played."


def test_split_sentences_multiple_punctuation():
    """Test splitting with different punctuation marks."""
    cfg = TextQualityConfig()
    text = "What happened? She was surprised! Then she left."
    sents = split_sentences(text, cfg)
    assert len(sents) == 3


def test_split_sentences_empty_text():
    """Test splitting empty or whitespace-only text."""
    cfg = TextQualityConfig()
    assert split_sentences("", cfg) == []
    assert split_sentences("   ", cfg) == []
    assert split_sentences(None, cfg) == []


def test_split_sentences_min_length_filter():
    """Test that very short sentences are filtered out."""
    cfg = TextQualityConfig(min_sentence_chars=5)
    text = "Hi. This is a longer sentence."
    sents = split_sentences(text, cfg)
    assert len(sents) == 1  # "Hi." is too short
    assert "longer" in sents[0]


def test_split_sentences_max_sentences_limit():
    """Test that max_sentences limit is respected."""
    cfg = TextQualityConfig(max_sentences=2)
    text = "First. Second. Third. Fourth."
    sents = split_sentences(text, cfg)
    assert len(sents) == 2


def test_split_sentences_whitespace_normalization():
    """Test that extra whitespace is normalized."""
    cfg = TextQualityConfig()
    text = "First   sentence.    Second\n\nsentence."
    sents = split_sentences(text, cfg)
    assert len(sents) == 2
    assert "  " not in sents[0]  # No double spaces


# =============================================================================
# Entity Extraction Tests
# =============================================================================

def test_fallback_entities_basic():
    """Test basic fallback entity extraction."""
    cfg = TextQualityConfig()
    sentence = "Lily went to the park with Tom."
    entities = _fallback_entities(sentence, cfg)
    
    assert "lily" in entities
    assert "park" in entities
    assert "tom" in entities
    # Stopwords should be filtered
    assert "the" not in entities
    assert "to" not in entities
    assert "with" not in entities


def test_fallback_entities_min_length():
    """Test that short entities are filtered."""
    cfg = TextQualityConfig(min_entity_len=4)
    sentence = "A cat sat on a mat."
    entities = _fallback_entities(sentence, cfg)
    
    # Only entities >= 4 chars
    assert "cat" not in entities  # 3 chars
    assert "sat" not in entities  # 3 chars
    assert "mat" not in entities  # 3 chars


def test_fallback_entities_contractions():
    """Test handling of contractions."""
    cfg = TextQualityConfig()
    sentence = "She's happy and he'll come."
    entities = _fallback_entities(sentence, cfg)
    
    # Contractions should be extracted as single tokens
    assert "happy" in entities
    assert "come" in entities


def test_fallback_entities_empty():
    """Test fallback entity extraction with empty or stopword-only text."""
    cfg = TextQualityConfig()
    assert len(_fallback_entities("", cfg)) == 0
    assert len(_fallback_entities("the and or", cfg)) == 0


def test_extract_entities_fallback_mode():
    """Test extract_entities using fallback (no spaCy)."""
    cfg = TextQualityConfig(use_spacy=False)
    sentences = ["Lily played.", "The park was fun."]
    entity_sets = extract_entities(sentences, cfg)
    
    assert len(entity_sets) == 2
    assert "lily" in entity_sets[0]
    assert "park" in entity_sets[1]
    assert "fun" in entity_sets[1]


# =============================================================================
# Jaccard Similarity Tests
# =============================================================================

def test_jaccard_identical_sets():
    """Test Jaccard with identical sets."""
    a = {"cat", "dog", "bird"}
    b = {"cat", "dog", "bird"}
    assert jaccard(a, b) == 1.0


def test_jaccard_disjoint_sets():
    """Test Jaccard with completely different sets."""
    a = {"cat", "dog"}
    b = {"bird", "fish"}
    assert jaccard(a, b) == 0.0


def test_jaccard_partial_overlap():
    """Test Jaccard with partial overlap."""
    a = {"cat", "dog", "bird"}
    b = {"dog", "bird", "fish"}
    # Intersection: {dog, bird} = 2
    # Union: {cat, dog, bird, fish} = 4
    assert jaccard(a, b) == 0.5


def test_jaccard_empty_sets():
    """Test Jaccard with empty sets."""
    assert jaccard(set(), set()) == 1.0  # Both empty = perfect match
    assert jaccard({"cat"}, set()) == 0.0
    assert jaccard(set(), {"dog"}) == 0.0


# =============================================================================
# Coherence Tests
# =============================================================================

def test_compute_coherence_basic():
    """Test basic coherence computation."""
    entity_sets = [
        {"lily", "park"},
        {"lily", "swing"},
        {"lily", "home"}
    ]
    result = compute_coherence_entity_overlap(entity_sets)
    
    assert "coherence" in result
    coherence = result["coherence"]
    assert 0.0 <= coherence <= 1.0
    assert coherence > 0.0  # "lily" appears in all


def test_compute_coherence_perfect():
    """Test coherence with perfect entity continuity."""
    entity_sets = [
        {"lily", "park"},
        {"lily", "park"}
    ]
    result = compute_coherence_entity_overlap(entity_sets)
    assert result["coherence"] == 1.0


def test_compute_coherence_zero():
    """Test coherence with no entity overlap."""
    entity_sets = [
        {"cat", "dog"},
        {"bird", "fish"}
    ]
    result = compute_coherence_entity_overlap(entity_sets)
    assert result["coherence"] == 0.0


def test_compute_coherence_single_sentence():
    """Test coherence with only one sentence."""
    entity_sets = [{"lily", "park"}]
    result = compute_coherence_entity_overlap(entity_sets)
    assert math.isnan(result["coherence"])


def test_compute_coherence_empty():
    """Test coherence with no sentences."""
    entity_sets = []
    result = compute_coherence_entity_overlap(entity_sets)
    assert math.isnan(result["coherence"])


# =============================================================================
# Cohesion Tests
# =============================================================================

@patch('pocket_narrator.text_quality._HAS_ST', False)
def test_compute_cohesion_without_st():
    """Test cohesion when sentence-transformers is not available."""
    cfg = TextQualityConfig(use_sentence_transformers=True)
    sentences = ["First sentence.", "Second sentence."]
    
    result = compute_cohesion_semantic(sentences, cfg)
    
    assert math.isnan(result["cohesion_mean"])
    assert math.isnan(result["cohesion_min"])
    assert math.isnan(result["cohesion_low_rate"])


def test_compute_cohesion_disabled():
    """Test cohesion when explicitly disabled."""
    cfg = TextQualityConfig(use_sentence_transformers=False)
    sentences = ["First sentence.", "Second sentence."]
    
    result = compute_cohesion_semantic(sentences, cfg)
    
    assert math.isnan(result["cohesion_mean"])


def test_compute_cohesion_single_sentence():
    """Test cohesion with only one sentence."""
    cfg = TextQualityConfig()
    sentences = ["Only one sentence."]
    
    result = compute_cohesion_semantic(sentences, cfg)
    
    assert math.isnan(result["cohesion_mean"])


@patch('pocket_narrator.text_quality._HAS_ST', True)
def test_compute_cohesion_with_mock_embedder():
    """Test cohesion computation with mocked embedder."""
    cfg = TextQualityConfig(use_sentence_transformers=True)
    sentences = ["Lily went to the park.", "Lily played on the swings."]
    
    # Create mock embedder
    mock_embedder = MagicMock()
    import numpy as np
    # High similarity vectors (nearly identical)
    mock_embedder.encode.return_value = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0]
    ])
    
    result = compute_cohesion_semantic(sentences, cfg, embedder=mock_embedder)
    
    assert "cohesion_mean" in result
    assert not math.isnan(result["cohesion_mean"])
    assert -1.0 <= result["cohesion_mean"] <= 1.0


# =============================================================================
# Final Quality Score Tests
# =============================================================================

def test_compute_final_quality_both_metrics():
    """Test final quality when both coherence and cohesion are available."""
    cfg = TextQualityConfig(w_coherence=0.6, w_cohesion=0.4)
    metrics = {
        "coherence": 0.8,
        "cohesion_mean": 0.6
    }
    
    quality = compute_final_quality(metrics, cfg)
    
    expected = 0.6 * 0.8 + 0.4 * 0.6  # 0.48 + 0.24 = 0.72
    assert abs(quality - expected) < 0.001


def test_compute_final_quality_only_coherence():
    """Test final quality when only coherence is available."""
    cfg = TextQualityConfig()
    metrics = {
        "coherence": 0.7,
        "cohesion_mean": float("nan")
    }
    
    quality = compute_final_quality(metrics, cfg)
    assert quality == 0.7


def test_compute_final_quality_only_cohesion():
    """Test final quality when only cohesion is available."""
    cfg = TextQualityConfig()
    metrics = {
        "coherence": float("nan"),
        "cohesion_mean": 0.8
    }
    
    quality = compute_final_quality(metrics, cfg)
    assert quality == 0.8


def test_compute_final_quality_neither():
    """Test final quality when neither metric is available."""
    cfg = TextQualityConfig()
    metrics = {
        "coherence": float("nan"),
        "cohesion_mean": float("nan")
    }
    
    quality = compute_final_quality(metrics, cfg)
    assert math.isnan(quality)


# =============================================================================
# End-to-End Evaluation Tests
# =============================================================================

def test_evaluate_text_quality_basic():
    """Test end-to-end evaluation with a simple story."""
    story = "Lily went to the park. She played on the swings. Then Lily went home."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert "n_sentences" in result
    assert result["n_sentences"] == 3
    assert "coherence" in result
    assert "cohesion_mean" in result
    assert "text_quality" in result
    assert not math.isnan(result["coherence"])  # Should compute coherence
    assert math.isnan(result["cohesion_mean"])  # ST disabled


def test_evaluate_text_quality_empty():
    """Test evaluation with empty text."""
    result = evaluate_text_quality("", cfg=TextQualityConfig())
    
    assert result["n_sentences"] == 0
    assert math.isnan(result["coherence"])


def test_evaluate_text_quality_single_sentence():
    """Test evaluation with only one sentence."""
    story = "This is a single sentence."
    cfg = TextQualityConfig(use_spacy=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert result["n_sentences"] == 1
    assert math.isnan(result["coherence"])  # Need >= 2 sentences


def test_evaluate_text_quality_high_coherence():
    """Test story with high entity coherence."""
    story = "Max the dog played. Max ran fast. Max was happy."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    # "max" appears in all sentences, creating some coherence
    assert result["coherence"] > 0.0
    assert result["coherence"] < 1.0


def test_evaluate_text_quality_low_coherence():
    """Test story with low entity coherence."""
    story = "The cat slept. Birds flew south. Cars drove by."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    # Different entities in each sentence
    assert result["coherence"] < 0.3


def test_evaluate_text_quality_custom_config():
    """Test evaluation with custom configuration."""
    story = "A cat. A dog. A bird."
    cfg = TextQualityConfig(
        min_sentence_chars=1,
        min_entity_len=3,
        w_coherence=0.7,
        w_cohesion=0.3,
        use_spacy=False,
        use_sentence_transformers=False
    )
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert result["n_sentences"] == 3
    assert "coherence" in result


def test_evaluate_text_quality_with_config_none():
    """Test that default config is used when cfg=None."""
    story = "First sentence. Second sentence."
    result = evaluate_text_quality(story, cfg=None)
    
    assert "n_sentences" in result
    assert result["n_sentences"] == 2


# =============================================================================
# Embedder Tests
# =============================================================================

def test_embedder_lazy_loading():
    """Test that embedder only loads model on first encode call."""
    embedder = _Embedder("test-model")
    
    # Model should not be loaded yet
    assert embedder._model is None
    
    # We can't easily test the actual loading without sentence-transformers installed,
    # but we can verify the lazy loading structure is in place
    assert embedder.model_name == "test-model"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_evaluate_text_quality_very_long_story():
    """Test with story exceeding max_sentences limit."""
    sentences = [f"Sentence number {i}." for i in range(100)]
    story = " ".join(sentences)
    cfg = TextQualityConfig(max_sentences=10, use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    # Should only process first 10 sentences
    assert result["n_sentences"] == 10


def test_evaluate_text_quality_special_characters():
    """Test with special characters and unicode."""
    story = "Café ☕ opened. The café was cozy. People loved it."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert result["n_sentences"] == 3
    assert not math.isnan(result["coherence"])


def test_evaluate_text_quality_numbers():
    """Test with numbers in text."""
    story = "Room 123 was empty. Room 456 was full. The rooms were different."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert result["n_sentences"] == 3


def test_evaluate_text_quality_repeated_content():
    """Test with highly repetitive content."""
    story = "The cat sat. The cat sat. The cat sat."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    # Should have perfect coherence (identical entities)
    assert result["coherence"] == 1.0


# =============================================================================
# Integration Tests
# =============================================================================

def test_evaluate_text_quality_realistic_story():
    """Test with a realistic TinyStories-style story."""
    story = """
    Once upon a time, there was a little girl named Lily. 
    Lily loved to play in the park with her friends.
    One day, Lily found a beautiful butterfly in the garden.
    She watched the butterfly fly from flower to flower.
    Lily was very happy and ran home to tell her mom.
    """
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert result["n_sentences"] == 5
    assert result["coherence"] > 0.0  # Some entity continuity
    assert not math.isnan(result["text_quality"])
    assert 0.0 <= result["text_quality"] <= 1.0


def test_avg_entities_per_sentence():
    """Test average entities per sentence calculation."""
    story = "The big cat sat. A small dog ran quickly."
    cfg = TextQualityConfig(use_spacy=False, use_sentence_transformers=False)
    
    result = evaluate_text_quality(story, cfg=cfg)
    
    assert "avg_entities_per_sentence" in result
    assert result["avg_entities_per_sentence"] > 0
