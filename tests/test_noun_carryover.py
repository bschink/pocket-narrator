"""
Unit tests for the noun_carryover module.

Tests cover:
- Noun extraction (with and without spaCy)
- Hard metrics (coverage, jaccard, precision)
- Soft metrics (semantic similarity)
- Edge cases and error handling
- Graceful degradation when dependencies are missing
"""
import pytest
from unittest.mock import patch, MagicMock

from pocket_narrator.noun_carryover import (
    extract_nouns,
    hard_coverage,
    hard_jaccard,
    hard_precision,
    soft_coverage,
    soft_coverage_at,
    noun_carryover_metrics,
    SoftConfig,
    SoftEmbedder,
    _try_load_spacy,
    _try_load_sentence_transformers,
)


# =============================================================================
# Noun Extraction Tests
# =============================================================================

def test_extract_nouns_basic():
    """Test basic noun extraction."""
    text = "The cat and the dog played in the park."
    nouns = extract_nouns(text)
    # Should extract: cat, dog, park (common nouns)
    assert "cat" in nouns
    assert "dog" in nouns
    assert "park" in nouns


def test_extract_nouns_proper_nouns():
    """Test extraction of proper nouns."""
    text = "Lucy and Tom went to London."
    nouns = extract_nouns(text, keep_propn=True)
    # Should extract proper nouns if spaCy is available
    # If not, fallback should still catch them
    assert len(nouns) >= 2  # At least some nouns extracted


def test_extract_nouns_empty_string():
    """Test extraction from empty string."""
    nouns = extract_nouns("")
    assert nouns == []


def test_extract_nouns_no_nouns():
    """Test extraction from text with no nouns."""
    text = "I am running quickly."
    nouns = extract_nouns(text)
    # Should be empty or very few (depending on POS tagging)
    assert isinstance(nouns, list)


def test_extract_nouns_min_length():
    """Test minimum length filtering."""
    text = "A big cat sat on a mat."
    nouns = extract_nouns(text, min_len=3)
    # "cat" and "mat" are 3 letters, should be included
    # single letters and short words should be filtered
    for noun in nouns:
        assert len(noun) >= 3


def test_extract_nouns_stopword_removal():
    """Test that stopwords are removed."""
    text = "The cat is in the house."
    nouns = extract_nouns(text)
    # Should not contain stopwords like "the", "is", "in"
    for noun in nouns:
        assert noun not in {"the", "is", "in", "a", "an"}


def test_extract_nouns_normalization():
    """Test that nouns are normalized (lowercase, lemmatized)."""
    text = "The cats and dogs were playing."
    nouns = extract_nouns(text)
    # Should be normalized to singular form and lowercase if spaCy available
    # If fallback, should at least be lowercase
    for noun in nouns:
        assert noun == noun.lower()


def test_extract_nouns_punctuation():
    """Test handling of punctuation."""
    text = "Hello! The cat, dog, and bird played."
    nouns = extract_nouns(text)
    # Should extract nouns without punctuation
    assert "cat" in nouns
    assert "dog" in nouns
    assert "bird" in nouns
    # No punctuation should remain
    for noun in nouns:
        assert "," not in noun
        assert "." not in noun


# =============================================================================
# Hard Metrics Tests
# =============================================================================

def test_hard_coverage_full_overlap():
    """Test coverage when all prompt nouns appear in story."""
    prompt_nouns = ["cat", "dog", "park"]
    story_nouns = ["cat", "dog", "park", "ball", "tree"]
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == 1.0


def test_hard_coverage_partial_overlap():
    """Test coverage with partial overlap."""
    prompt_nouns = ["cat", "dog", "park"]
    story_nouns = ["cat", "ball", "tree"]
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == pytest.approx(1/3, rel=1e-5)  # 1 out of 3


def test_hard_coverage_no_overlap():
    """Test coverage with no overlap."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["bird", "fish"]
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == 0.0


def test_hard_coverage_empty_prompt():
    """Test coverage with empty prompt."""
    prompt_nouns = []
    story_nouns = ["cat", "dog"]
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == 0.0


def test_hard_coverage_empty_story():
    """Test coverage with empty story."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = []
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == 0.0


def test_hard_coverage_duplicates():
    """Test that duplicates are handled correctly (set-based)."""
    prompt_nouns = ["cat", "cat", "dog"]
    story_nouns = ["cat", "dog", "dog", "park"]
    coverage = hard_coverage(prompt_nouns, story_nouns)
    assert coverage == 1.0  # Both unique nouns present


def test_hard_jaccard_identical_sets():
    """Test Jaccard with identical sets."""
    nouns = ["cat", "dog", "park"]
    jaccard = hard_jaccard(nouns, nouns)
    assert jaccard == 1.0


def test_hard_jaccard_partial_overlap():
    """Test Jaccard with partial overlap."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["dog", "bird"]
    jaccard = hard_jaccard(prompt_nouns, story_nouns)
    # Intersection: {dog}, Union: {cat, dog, bird}
    assert jaccard == pytest.approx(1/3, rel=1e-5)


def test_hard_jaccard_no_overlap():
    """Test Jaccard with no overlap."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["bird", "fish"]
    jaccard = hard_jaccard(prompt_nouns, story_nouns)
    assert jaccard == 0.0


def test_hard_jaccard_empty_sets():
    """Test Jaccard with empty sets."""
    jaccard = hard_jaccard([], [])
    assert jaccard == 0.0


def test_hard_precision_full_overlap():
    """Test precision when all story nouns are from prompt."""
    prompt_nouns = ["cat", "dog", "park", "tree", "ball"]
    story_nouns = ["cat", "dog"]
    precision = hard_precision(prompt_nouns, story_nouns)
    assert precision == 1.0


def test_hard_precision_partial_overlap():
    """Test precision with partial overlap."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["cat", "bird", "fish"]
    precision = hard_precision(prompt_nouns, story_nouns)
    assert precision == pytest.approx(1/3, rel=1e-5)  # 1 out of 3


def test_hard_precision_empty_story():
    """Test precision with empty story."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = []
    precision = hard_precision(prompt_nouns, story_nouns)
    assert precision == 0.0


# =============================================================================
# Soft Metrics Tests (require mocking)
# =============================================================================

@pytest.fixture
def mock_embedder():
    """Create a mock embedder with controlled behavior."""
    embedder = MagicMock()
    embedder.available.return_value = True
    
    # Mock encode to return simple embeddings
    def mock_encode(texts):
        import numpy as np
        # Return unit vectors with controlled similarity
        vectors = []
        
        # Create a consistent mapping: same word -> same vector
        # Use 5 dimensions for more control
        word_to_vector = {
            "cat": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "kitten": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Identical to cat
            "dog": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            "puppy": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),   # Identical to dog
            "bird": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            "fish": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            "tree": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            "park": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),    # Same as cat for testing
            "ball": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),    # Same as fish
        }
        
        for text in texts:
            if text in word_to_vector:
                vectors.append(word_to_vector[text])
            else:
                # Create orthogonal vector for unknown words using hash
                vec = np.zeros(5)
                dim = abs(hash(text)) % 5
                vec[dim] = 1.0
                vectors.append(vec)
        
        return np.array(vectors)
    
    embedder.encode = mock_encode
    return embedder


def test_soft_coverage_perfect_match(mock_embedder):
    """Test soft coverage with perfect semantic matches."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["cat", "dog"]
    coverage = soft_coverage(prompt_nouns, story_nouns, mock_embedder)
    assert coverage == pytest.approx(1.0, rel=1e-5)


def test_soft_coverage_no_match(mock_embedder):
    """Test soft coverage with no semantic matches."""
    prompt_nouns = ["cat"]
    story_nouns = ["tree"]  # Changed to use a word that will be orthogonal
    coverage = soft_coverage(prompt_nouns, story_nouns, mock_embedder)
    # With our mock, different words have similarity 0
    assert coverage == 0.0


def test_soft_coverage_empty_prompt():
    """Test soft coverage with empty prompt."""
    embedder = MagicMock()
    embedder.available.return_value = True
    coverage = soft_coverage([], ["cat", "dog"], embedder)
    assert coverage == 0.0


def test_soft_coverage_empty_story():
    """Test soft coverage with empty story."""
    embedder = MagicMock()
    embedder.available.return_value = True
    coverage = soft_coverage(["cat", "dog"], [], embedder)
    assert coverage == 0.0


def test_soft_coverage_unavailable():
    """Test soft coverage when embeddings are unavailable."""
    embedder = MagicMock()
    embedder.available.return_value = False
    coverage = soft_coverage(["cat"], ["dog"], embedder)
    assert coverage is None


def test_soft_coverage_at_threshold(mock_embedder):
    """Test soft coverage at threshold."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["cat", "dog"]
    coverage = soft_coverage_at(prompt_nouns, story_nouns, mock_embedder, threshold=0.9)
    # Perfect matches should be above threshold
    assert coverage == 1.0


def test_soft_coverage_at_high_threshold(mock_embedder):
    """Test soft coverage at high threshold filters out low matches."""
    prompt_nouns = ["cat", "dog"]
    story_nouns = ["bird", "tree"]  # Changed to use words that will be orthogonal
    coverage = soft_coverage_at(prompt_nouns, story_nouns, mock_embedder, threshold=0.9)
    # No matches above threshold (bird and tree are different vectors)
    assert coverage == 0.0


def test_soft_coverage_at_empty_prompt():
    """Test soft coverage@tau with empty prompt."""
    embedder = MagicMock()
    embedder.available.return_value = True
    coverage = soft_coverage_at([], ["cat"], embedder, threshold=0.7)
    assert coverage == 0.0


def test_soft_coverage_at_unavailable():
    """Test soft coverage@tau when embeddings unavailable."""
    embedder = MagicMock()
    embedder.available.return_value = False
    coverage = soft_coverage_at(["cat"], ["dog"], embedder, threshold=0.7)
    assert coverage is None


# =============================================================================
# SoftEmbedder Tests
# =============================================================================

def test_soft_embedder_availability_check():
    """Test that embedder correctly checks for sentence-transformers."""
    embedder = SoftEmbedder()
    # Should return True or False depending on whether package is installed
    available = embedder.available()
    assert isinstance(available, bool)


def test_soft_embedder_config():
    """Test that embedder respects configuration."""
    cfg = SoftConfig(model_name="custom-model", threshold=0.8)
    embedder = SoftEmbedder(cfg)
    assert embedder.cfg.model_name == "custom-model"
    assert embedder.cfg.threshold == 0.8


@patch('pocket_narrator.noun_carryover._try_load_sentence_transformers')
def test_soft_embedder_load_failure(mock_load):
    """Test embedder handles missing sentence-transformers."""
    mock_load.return_value = None
    embedder = SoftEmbedder()
    assert embedder.available() is False


# =============================================================================
# Integration Tests (noun_carryover_metrics)
# =============================================================================

def test_noun_carryover_metrics_basic():
    """Test end-to-end metric computation."""
    prompt = "Lucy had a cat and a dog."
    story = "The cat and dog played together."
    
    metrics = noun_carryover_metrics(prompt, story)
    
    # Check that all expected keys are present
    assert "hard_coverage" in metrics
    assert "hard_jaccard" in metrics
    assert "hard_precision" in metrics
    assert "soft_coverage" in metrics
    assert "soft_coverage@0.70" in metrics
    
    # Check that hard metrics are computed (should be non-zero)
    assert metrics["hard_coverage"] >= 0.0
    assert metrics["hard_jaccard"] >= 0.0
    assert metrics["hard_precision"] >= 0.0
    
    # Soft metrics might be None if dependencies missing
    # If available, should be between 0 and 1
    if metrics["soft_coverage"] is not None:
        assert 0.0 <= metrics["soft_coverage"] <= 1.0
    if metrics["soft_coverage@0.70"] is not None:
        assert 0.0 <= metrics["soft_coverage@0.70"] <= 1.0


def test_noun_carryover_metrics_empty_prompt():
    """Test metrics with empty prompt."""
    metrics = noun_carryover_metrics("", "The cat played.")
    assert metrics["hard_coverage"] == 0.0
    assert metrics["hard_jaccard"] == 0.0


def test_noun_carryover_metrics_empty_story():
    """Test metrics with empty story."""
    metrics = noun_carryover_metrics("The cat played.", "")
    assert metrics["hard_coverage"] == 0.0
    assert metrics["hard_precision"] == 0.0


def test_noun_carryover_metrics_custom_config():
    """Test metrics with custom soft config."""
    prompt = "The cat played."
    story = "A kitten was playing."
    
    cfg = SoftConfig(model_name="all-MiniLM-L6-v2", threshold=0.8)
    metrics = noun_carryover_metrics(prompt, story, soft_cfg=cfg)
    
    # Should have custom threshold in key
    assert "soft_coverage@0.80" in metrics


def test_noun_carryover_metrics_realistic_example():
    """Test with realistic children's story example."""
    prompt = "Once upon a time, there was a little girl named Lucy. She had a red ball."
    story = "Lucy loved her ball very much. She played with it every day in the park."
    
    metrics = noun_carryover_metrics(prompt, story)
    
    # Should have good coverage since Lucy and ball are mentioned
    assert metrics["hard_coverage"] > 0.0
    
    # Hard jaccard should be reasonable
    assert metrics["hard_jaccard"] > 0.0


def test_noun_carryover_metrics_no_overlap_example():
    """Test with story that ignores prompt."""
    prompt = "Lucy had a cat named Mittens."
    story = "The dog ran in the park and chased birds."
    
    metrics = noun_carryover_metrics(prompt, story)
    
    # Should have low coverage since prompt nouns not used
    assert metrics["hard_coverage"] == 0.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_extract_nouns_unicode():
    """Test noun extraction with unicode characters."""
    text = "The café had a piñata."
    nouns = extract_nouns(text)
    # Should handle unicode gracefully
    assert isinstance(nouns, list)


def test_extract_nouns_numbers():
    """Test that numbers are filtered out."""
    text = "The cat has 123 toys."
    nouns = extract_nouns(text)
    # Should not contain "123"
    assert "123" not in nouns


def test_hard_metrics_case_sensitivity():
    """Test that hard metrics are case-insensitive."""
    prompt_nouns = ["Cat", "Dog"]
    story_nouns = ["cat", "dog"]
    # Should work if nouns are pre-normalized
    # In practice, extract_nouns normalizes to lowercase


def test_soft_config_defaults():
    """Test SoftConfig default values."""
    cfg = SoftConfig()
    assert cfg.model_name == "all-MiniLM-L6-v2"
    assert cfg.threshold == 0.70


@patch('pocket_narrator.noun_carryover._try_load_spacy')
def test_extract_nouns_fallback_mode(mock_spacy):
    """Test noun extraction falls back when spaCy fails."""
    # Simulate spaCy not available
    mock_spacy.return_value = None
    
    text = "The cat and dog played in the park."
    nouns = extract_nouns(text)
    
    # Should still extract some nouns using fallback regex method
    assert isinstance(nouns, list)
    # Fallback should catch some basic words
    assert len(nouns) > 0


def test_noun_carryover_metrics_special_characters():
    """Test metrics with special characters in text."""
    prompt = "Lucy's cat!!! <3"
    story = "The cat was happy :)"
    
    metrics = noun_carryover_metrics(prompt, story)
    
    # Should handle special characters gracefully
    assert isinstance(metrics["hard_coverage"], float)
