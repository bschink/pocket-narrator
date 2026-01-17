# text_quality.py with zero changes to training code. We only need the generations JSONL (prompt/story/model_id)...
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Optional deps
_HAS_SPACY = False
_HAS_ST = False

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

STOPWORDS_FALLBACK = {
    # minimal English stop list (TinyStories)
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because",
    "to", "of", "in", "on", "at", "for", "with", "from", "into", "over",
    "is", "was", "were", "are", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "hers", "our", "their",
    "this", "that", "these", "those",
    "as", "by", "not", "no", "yes", "do", "did", "does",
    "very", "just", "really", "too", "also",
}


@dataclass
class TextQualityConfig:
    # Sentence splitting
    min_sentence_chars: int = 2
    max_sentences: int = 50

    # Entity extraction
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    keep_entity_pos: Tuple[str, ...] = ("NOUN", "PROPN")  # TinyStories focus 
    min_entity_len: int = 2

    # Cohesion embeddings
    use_sentence_transformers: bool = True
    st_model: str = "all-MiniLM-L6-v2"

    # Threshold for "topic shift" detection
    cohesion_tau: float = 0.35

    # Final score weights
    w_coherence: float = 0.6
    w_cohesion: float = 0.4


class _Embedder:
    """Lazy embedder; only loads a SentenceTransformer if available and requested."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, sentences: List[str]):
        if not _HAS_ST:
            raise RuntimeError("sentence_transformers not installed. Install with: pip install sentence-transformers")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(sentences, normalize_embeddings=True)


def split_sentences(text: str, cfg: TextQualityConfig) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    sents = _SENT_SPLIT_RE.split(text)
    sents = [s.strip() for s in sents if len(s.strip()) >= cfg.min_sentence_chars]
    return sents[: cfg.max_sentences]


def _fallback_entities(sentence: str, cfg: TextQualityConfig) -> Set[str]:
    # Heuristic: keep alphabetic tokens, lowercase, remove stopwords
    toks = _TOKEN_RE.findall(sentence)
    ents = set()
    for t in toks:
        tl = t.lower()
        if len(tl) < cfg.min_entity_len:
            continue
        if tl in STOPWORDS_FALLBACK:
            continue
        ents.add(tl)
    return ents


def _spacy_entities(nlp, sentence: str, cfg: TextQualityConfig) -> Set[str]:
    doc = nlp(sentence)
    ents: Set[str] = set()
    for token in doc:
        if token.pos_ not in cfg.keep_entity_pos:
            continue
        tl = token.lemma_.lower().strip()
        if len(tl) < cfg.min_entity_len:
            continue
        if tl in STOPWORDS_FALLBACK:
            continue
        ents.add(tl)
    return ents


def extract_entities(sentences: List[str], cfg: TextQualityConfig):
    """
    Returns list of entity sets per sentence, using spaCy if available/requested,
    otherwise heuristic fallback.
    """
    if cfg.use_spacy and _HAS_SPACY:
        try:
            nlp = spacy.load(cfg.spacy_model, disable=["ner", "textcat"])
            return [_spacy_entities(nlp, s, cfg) for s in sentences]
        except Exception:
            # fall back silently
            return [_fallback_entities(s, cfg) for s in sentences]
    return [_fallback_entities(s, cfg) for s in sentences]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def compute_coherence_entity_overlap(entity_sets: List[Set[str]]) -> Dict[str, float]:
    """
    TinyStories coherence = average adjacent-sentence entity Jaccard overlap.
    """
    n = len(entity_sets)
    if n < 2:
        return {"coherence": float("nan")}
    overlaps = [jaccard(entity_sets[i], entity_sets[i + 1]) for i in range(n - 1)]
    return {"coherence": sum(overlaps) / len(overlaps) if overlaps else float("nan")}


def _cosine(u, v) -> float:
    # assumes already normalized if using sentence-transformers encode(normalize_embeddings=True)
    s = float((u * v).sum())
    # clamp due to numerical issues
    return max(-1.0, min(1.0, s))


def compute_cohesion_semantic(sentences: List[str], cfg: TextQualityConfig, embedder: Optional[_Embedder] = None) -> Dict[str, float]:
    """
    TinyStories cohesion = sentence embedding cosine similarity of adjacent sentences.
    Returns mean, min, and low-cohesion rate (below cfg.cohesion_tau).
    If sentence-transformers unavailable, returns NaNs.
    """
    n = len(sentences)
    if n < 2:
        return {"cohesion_mean": float("nan"), "cohesion_min": float("nan"), "cohesion_low_rate": float("nan")}

    if not (cfg.use_sentence_transformers and _HAS_ST):
        return {"cohesion_mean": float("nan"), "cohesion_min": float("nan"), "cohesion_low_rate": float("nan")}

    if embedder is None:
        embedder = _Embedder(cfg.st_model)

    vecs = embedder.encode(sentences)
    sims: List[float] = []
    for i in range(n - 1):
        sims.append(_cosine(vecs[i], vecs[i + 1]))

    mean_sim = sum(sims) / len(sims) if sims else float("nan")
    min_sim = min(sims) if sims else float("nan")
    low_rate = sum(1 for s in sims if s < cfg.cohesion_tau) / len(sims) if sims else float("nan")

    return {"cohesion_mean": mean_sim, "cohesion_min": min_sim, "cohesion_low_rate": low_rate}


def compute_final_quality(metrics: Dict[str, float], cfg: TextQualityConfig) -> float:
    coh = metrics.get("coherence", float("nan"))
    cos = metrics.get("cohesion_mean", float("nan"))
    if math.isnan(coh) and math.isnan(cos):
        return float("nan")
    if math.isnan(coh):
        return cos
    if math.isnan(cos):
        return coh
    return cfg.w_coherence * coh + cfg.w_cohesion * cos


def evaluate_text_quality(story_text: str, cfg: Optional[TextQualityConfig] = None, embedder: Optional[_Embedder] = None) -> Dict[str, float]:
    """
    End-to-end TinyStories text quality evaluation:
      - coherence via entity overlap
      - cohesion via semantic similarity of adjacent sentences
      - final weighted score
    """
    cfg = cfg or TextQualityConfig()
    sents = split_sentences(story_text, cfg)
    entity_sets = extract_entities(sents, cfg)

    out: Dict[str, float] = {
        "n_sentences": float(len(sents)),
        "avg_entities_per_sentence": float(sum(len(es) for es in entity_sets) / len(entity_sets)) if entity_sets else float("nan"),
    }
    out.update(compute_coherence_entity_overlap(entity_sets))
    out.update(compute_cohesion_semantic(sents, cfg, embedder=embedder))
    out["text_quality"] = compute_final_quality(out, cfg)
    return out
