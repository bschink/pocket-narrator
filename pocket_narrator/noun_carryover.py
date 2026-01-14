"""

Noun carryover metrics for prompt vs generated story.

Metrics:
  - hard_coverage: |P ∩ S| / |P|
  - hard_jaccard:  |P ∩ S| / |P ∪ S|
  - soft_coverage: avg_p max_s cos(emb(p), emb(s))
  - soft_coverage@tau: fraction of prompt nouns with max similarity >= tau

Dependencies:
  - spaCy + en_core_web_sm for POS tagging (NOUN/PROPN)
  - sentence-transformers for soft similarity cosine-similarity) matching

If deps are missing, the code degrades gracefully:
  - no spaCy -> naive tokenization heuristic (lower quality)
  - no sentence-transformers -> soft metrics return None


p_emb = embedder.encode(p)   # prompt nouns
s_emb = embedder.encode(s)   # story nouns
sim = p_emb @ s_emb.T        # cosine similarity (normalized embeddings)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import re


# Helpers: safe imports

def _try_load_spacy():
    try:
        import spacy  # type: ignore
        return spacy
    except Exception:
        return None

def _try_load_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception:
        return None


# Noun extraction

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # simple English-ish tokens


def extract_nouns(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    keep_propn: bool = True,
    min_len: int = 2,
) -> List[str]:
    """
    Extract normalized nouns from text.

    Normalization:
      - lowercase
      - lemma (if spaCy available)
    Filtering:
      - drop stopwords (if spaCy available)
      - drop non-alpha tokens (spaCy) / regex tokenization (fallback)
      - drop tokens shorter than min_len
    """
    text = (text or "").strip()
    if not text:
        return []

    spacy = _try_load_spacy()
    if spacy is not None:
        try:
            nlp = spacy.load(spacy_model)  # requires: python -m spacy download en_core_web_sm
            doc = nlp(text)
            out: List[str] = []
            allowed = {"NOUN"}
            if keep_propn:
                allowed.add("PROPN")

            for tok in doc:
                if tok.pos_ not in allowed:
                    continue
                if tok.is_stop:
                    continue
                if not tok.is_alpha:
                    continue
                lemma = (tok.lemma_ or tok.text).lower().strip()
                if len(lemma) < min_len:
                    continue
                out.append(lemma)
            return out
        except Exception:
            # Fall through to heuristic extraction
            pass

    # Fallback: heuristic “noun-ish” extraction (not true POS)
    # We keep alphabetic tokens and remove a tiny stopword list.
    # This is intentionally minimal; prefer spaCy in serious eval.
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with",
        "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
        "i", "you", "he", "she", "they", "we", "my", "your", "his", "her", "their", "our",
    }
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    toks = [t for t in toks if len(t) >= min_len and t not in stop]
    return toks



# Hard metrics

def hard_coverage(prompt_nouns: Sequence[str], story_nouns: Sequence[str]) -> float:
    """|P ∩ S| / |P| over distinct nouns."""
    pset: Set[str] = set(prompt_nouns)
    if not pset:
        return 0.0
    sset: Set[str] = set(story_nouns)
    return len(pset & sset) / float(len(pset))


def hard_jaccard(prompt_nouns: Sequence[str], story_nouns: Sequence[str]) -> float:
    """|P ∩ S| / |P ∪ S| over distinct nouns."""
    pset: Set[str] = set(prompt_nouns)
    sset: Set[str] = set(story_nouns)
    union = pset | sset
    if not union:
        return 0.0
    return len(pset & sset) / float(len(union))


def hard_precision(prompt_nouns: Sequence[str], story_nouns: Sequence[str]) -> float:
    """|P ∩ S| / |S| over distinct nouns (optional extra)."""
    sset: Set[str] = set(story_nouns)
    if not sset:
        return 0.0
    pset: Set[str] = set(prompt_nouns)
    return len(pset & sset) / float(len(sset))



# Soft metrics (embeddings)

@dataclass
class SoftConfig:
    model_name: str = "all-MiniLM-L6-v2"
    threshold: float = 0.70  # for soft_coverage@tau


class SoftEmbedder:
    """
    Small wrapper to lazily load a SentenceTransformer model.
    If sentence-transformers is not installed, embedding is unavailable.
    """
    def __init__(self, cfg: SoftConfig = SoftConfig()):
        self.cfg = cfg
        self._model = None

    def available(self) -> bool:
        return _try_load_sentence_transformers() is not None

    def _load(self):
        if self._model is None:
            SentenceTransformer = _try_load_sentence_transformers()
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed")
            self._model = SentenceTransformer(self.cfg.model_name)

    def encode(self, texts: List[str]):
        self._load()
        # SentenceTransformer returns numpy arrays by default
        return self._model.encode(texts, normalize_embeddings=True)


def _cosine_matrix(a, b):
    """
    Given normalized embeddings, cosine = dot product.
    a: [P, D], b: [S, D]
    returns: [P, S]
    """
    # works with numpy arrays
    return a @ b.T


def soft_coverage(
    prompt_nouns: Sequence[str],
    story_nouns: Sequence[str],
    embedder: SoftEmbedder,
) -> Optional[float]:
    """
    Average over prompt nouns of max cosine similarity to any story noun.
    Returns None if embeddings are unavailable or inputs are empty.
    """
    p = sorted(set(prompt_nouns))
    s = sorted(set(story_nouns))
    if not p or not s:
        return 0.0 if p else 0.0  # keep numeric stability for aggregation

    if not embedder.available():
        return None

    p_emb = embedder.encode(p)
    s_emb = embedder.encode(s)
    sim = _cosine_matrix(p_emb, s_emb)  # [P, S]
    # row-wise max, then mean
    import numpy as np  # local import to keep top-level deps minimal
    row_max = np.max(sim, axis=1)
    return float(np.mean(row_max))


def soft_coverage_at(
    prompt_nouns: Sequence[str],
    story_nouns: Sequence[str],
    embedder: SoftEmbedder,
    *,
    threshold: float,
) -> Optional[float]:
    """
    Fraction of prompt nouns whose best story match has cosine >= threshold.
    Returns None if embeddings are unavailable.
    """
    p = sorted(set(prompt_nouns))
    s = sorted(set(story_nouns))
    if not p:
        return 0.0
    if not s:
        return 0.0

    if not embedder.available():
        return None

    p_emb = embedder.encode(p)
    s_emb = embedder.encode(s)
    sim = _cosine_matrix(p_emb, s_emb)
    import numpy as np
    row_max = np.max(sim, axis=1)
    return float(np.mean(row_max >= threshold))



# One-shot compute function

def noun_carryover_metrics(
    prompt: str,
    story: str,
    *,
    spacy_model: str = "en_core_web_sm",
    soft_cfg: SoftConfig = SoftConfig(),
) -> Dict[str, Optional[float]]:
    """
    Convenience function to compute all metrics from raw strings.
    """
    p_nouns = extract_nouns(prompt, spacy_model=spacy_model)
    s_nouns = extract_nouns(story, spacy_model=spacy_model)

    out: Dict[str, Optional[float]] = {
        "hard_coverage": hard_coverage(p_nouns, s_nouns),
        "hard_jaccard": hard_jaccard(p_nouns, s_nouns),
        "hard_precision": hard_precision(p_nouns, s_nouns),
    }

    emb = SoftEmbedder(soft_cfg)
    out["soft_coverage"] = soft_coverage(p_nouns, s_nouns, emb)
    out[f"soft_coverage@{soft_cfg.threshold:.2f}"] = soft_coverage_at(
        p_nouns, s_nouns, emb, threshold=soft_cfg.threshold
    )
    return out
