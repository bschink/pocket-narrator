# mamba_model_compare.py
from __future__ import annotations
from typing import Dict, Any, List

import json
from dataclasses import dataclass


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, Any]


def load_results(paths: Dict[str, str]) -> List[ModelResult]:
    """
    paths: { model_name: path_to_metrics_json }
    JSON-Format z.B.:
      {
        "perplexity": 15.3,
        "bleu": 23.4,
        "rougeL": 0.31,
        "bertscore_f1": 0.81
      }
    """
    results = []
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        results.append(ModelResult(name=name, metrics=metrics))
    return results


def format_table(results: List[ModelResult], keys: List[str]) -> str:
    # simple ASCII-Tabelle
    header = ["model"] + keys
    rows = [header]

    for r in results:
        row = [r.name]
        for k in keys:
            v = r.metrics.get(k, None)
            if isinstance(v, float):
                row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        rows.append(row)

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]

    def fmt_row(row):
        return " | ".join(
            row[i].ljust(col_widths[i]) for i in range(len(row))
        )

    lines = [fmt_row(rows[0])]
    lines.append("-+-".join("-" * w for w in col_widths))
    lines.extend(fmt_row(r) for r in rows[1:])
    return "\n".join(lines)


if __name__ == "__main__":
    
    paths = {
        "gpt2_small": "results/gpt2_small/metrics.json",
        "gptneo_8M": "results/gptneo_8M/metrics.json",
        "mamba_8M": "results/mamba_8M/metrics.json",
    }
    results = load_results(paths)
    keys = ["perplexity", "bleu", "rougeL", "bertscore_f1"]
    print(format_table(results, keys))
