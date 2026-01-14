# this code with spaCy + en_core_web_sm for POS tagging (NOUN/PROPN) sentence-transformers for soft similarity
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pocket_narrator.noun_carryover import noun_carryover_metrics, SoftConfig


"""
--- python -m pip install spacy sentence-transformers
    python -m spacy download en_core_web

PYTHONPATH=. python3 scripts/evaluate.py \
  scripts/eval_noun_carryover.py \
  --in_jsonl runs/generations.jsonl \
  --out_dir runs/eval \
  --spacy_model en_core_web_sm \
  --soft_model all-MiniLM-L6-v2 \
  --soft_threshold 0.7
  


Evaluate prompt-vs-story noun carryover metrics on a JSONL file.

Input JSONL (one per line):
  {"model_id": "...", "prompt": "...", "story": "...", ...}

Outputs:
  - <out_dir>/per_sample.jsonl
  - <out_dir>/leaderboard.csv

This script is standalone and does not assume any training internals.

"""
"""
p_emb = embedder.encode(p)   # prompt nouns
s_emb = embedder.encode(s)   # story nouns
sim = p_emb @ s_emb.T        # cosine similarity (normalized embeddings)

"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True, help="JSONL with model_id/prompt/story")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    ap.add_argument("--soft_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--soft_threshold", type=float, default=0.70)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    soft_cfg = SoftConfig(model_name=args.soft_model, threshold=args.soft_threshold)

    data = read_jsonl(in_path)
    per_sample: List[Dict[str, Any]] = []

    by_model: Dict[str, Dict[str, List[Optional[float]]]] = defaultdict(lambda: defaultdict(list))

    for row in data:
        model_id = str(row.get("model_id", "unknown"))
        prompt = str(row.get("prompt", ""))
        story = str(row.get("story", ""))

        metrics = noun_carryover_metrics(
            prompt,
            story,
            spacy_model=args.spacy_model,
            soft_cfg=soft_cfg,
        )

        out_row = dict(row)
        out_row.update(metrics)
        per_sample.append(out_row)

        for k, v in metrics.items():
            by_model[model_id][k].append(v)

    write_jsonl(out_dir / "per_sample.jsonl", per_sample)

    # leaderboard
    metric_keys = [
        "hard_coverage",
        "hard_jaccard",
        "hard_precision",
        "soft_coverage",
        f"soft_coverage@{soft_cfg.threshold:.2f}",
    ]

    with (out_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "n_samples"] + metric_keys)

        for model_id, buckets in sorted(by_model.items(), key=lambda x: x[0]):
            n = len(per_sample) if model_id == "unknown" else len(buckets["hard_coverage"])
            row_out: List[Any] = [model_id, n]
            for mk in metric_keys:
                row_out.append(safe_mean(buckets.get(mk, [])))
            w.writerow(row_out)


if __name__ == "__main__":
    main()
