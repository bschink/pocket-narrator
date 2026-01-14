
"""
This script evaluates the *text quality* of TinyStories-style generations.

It reads a JSONL file where each line contains:
  {
    "model_id": "...",
    "prompt": "...",
    "story": "..."
  }

For each generated story, it computes:
  - Coherence   (entity continuity across sentences)
  - Cohesion    (semantic similarity between adjacent sentences)
  - Text quality (weighted combination of coherence + cohesion)

It produces two outputs:
  1) per_sample.jsonl  -> metrics for every story
  2) leaderboard.csv   -> averaged metrics per model_id

This script is evaluation-only:
  - It does NOT train models
  - It does NOT modify generation

"""
"""
python scripts/eval_text_quality.py \
  --input_jsonl runs/generations.jsonl \
  --out_dir runs/text_quality
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from pocket_narrator.text_quality import TextQualityConfig, evaluate_text_quality, _Embedder


def _safe_mean(xs: List[float]) -> float:
    """
    Compute the mean of a list while safely ignoring NaN values.

    Why?
    ----
    Some stories are too short (e.g. only one sentence),
    so coherence or cohesion cannot be computed and becomes NaN.

    We do not want NaNs to break model-level averages.
    """
    xs2 = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs2) / len(xs2) if xs2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True, help="JSONL with {model_id,prompt,story} per line")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--no_spacy", action="store_true", help="Disable spaCy entity extraction") # spaCy: nlp(sentence)
    ap.add_argument("--no_st", action="store_true", help="Disable sentence-transformers cohesion")
    
    """
    For decoder-only Transformers, all-MiniLM-L6-v2 is used as an external encoder to measure semantic cohesion between generated sentences, 
    ensuring architecture-independent and reproducible evaluation.

    """
    ap.add_argument("--st_model", type=str, default="all-MiniLM-L6-v2") # pretrained sentence-transformer model designed for semantic similarity tasks and best cost–quality tradeoff.
    ap.add_argument("--cohesion_tau", type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TextQualityConfig(
        use_spacy=not args.no_spacy,
        use_sentence_transformers=not args.no_st,
        st_model=args.st_model,
        cohesion_tau=args.cohesion_tau,
    )

    embedder = None
    if cfg.use_sentence_transformers:
        # one shared embedder for speed
        try:
            embedder = _Embedder(cfg.st_model)
        except Exception:
            embedder = None

    per_sample_path = out_dir / "per_sample.jsonl"
    leaderboard_path = out_dir / "leaderboard.csv"

    # Aggregate per model_id
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, open(per_sample_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            story = row.get("story", "") or ""
            model_id = row.get("model_id", "unknown")

            metrics = evaluate_text_quality(story, cfg=cfg, embedder=embedder)

            # Store metrics into row
            row.update({
                "n_sentences": metrics.get("n_sentences"),
                "avg_entities_per_sentence": metrics.get("avg_entities_per_sentence"),
                "coherence": metrics.get("coherence"),
                "cohesion_mean": metrics.get("cohesion_mean"),
                "cohesion_min": metrics.get("cohesion_min"),
                "cohesion_low_rate": metrics.get("cohesion_low_rate"),
                "text_quality": metrics.get("text_quality"),
            })

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Aggregate
            for k in ["coherence", "cohesion_mean", "cohesion_min", "cohesion_low_rate", "text_quality", "n_sentences", "avg_entities_per_sentence"]:
                v = row.get(k)
                if v is None:
                    continue
                try:
                    agg[model_id][k].append(float(v))
                except Exception:
                    pass

    # Write leaderboard
    fieldnames = [
        "model_id",
        "n_samples",
        "coherence_mean",
        "cohesion_mean_mean",
        "cohesion_min_mean",
        "cohesion_low_rate_mean",
        "text_quality_mean",
        "n_sentences_mean",
        "avg_entities_per_sentence_mean",
    ]
    # Write model-level leaderboard
    with open(leaderboard_path, "w", encoding="utf-8", newline="") as f_csv:
        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()
        for model_id, md in sorted(agg.items(), key=lambda x: x[0]):
            n = len(md.get("text_quality", [])) or max((len(v) for v in md.values()), default=0)
            w.writerow({
                "model_id": model_id,
                "n_samples": n,
                "coherence_mean": _safe_mean(md.get("coherence", [])),
                "cohesion_mean_mean": _safe_mean(md.get("cohesion_mean", [])),
                "cohesion_min_mean": _safe_mean(md.get("cohesion_min", [])),
                "cohesion_low_rate_mean": _safe_mean(md.get("cohesion_low_rate", [])),
                "text_quality_mean": _safe_mean(md.get("text_quality", [])),
                "n_sentences_mean": _safe_mean(md.get("n_sentences", [])),
                "avg_entities_per_sentence_mean": _safe_mean(md.get("avg_entities_per_sentence", [])),
            })


if __name__ == "__main__":
    main()


