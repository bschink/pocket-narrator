"""
Dataset evaluation script for the PocketNarrator project.

This script runs evaluation metrics on a dataset to analyze its quality,
diversity, and linguistic properties.

Usage:
    PYTHONPATH=. python3 scripts/evaluate_dataset.py \
      --data data/processed/TinyStories/TinyStories-train.bos_eos.txt \
      --check_grammar
      --batch_size 512

    PYTHONPATH=. python3 scripts/evaluate_dataset.py \
      --data data/raw/TinyStories/TinyStories-train.txt \
      --no_check_grammar \
      --sample_size 10000
      --batch_size 256
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime

from pocket_narrator.evaluate import run_dataset_evaluation
from pocket_narrator.data_loader import load_text_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a text dataset for quality and diversity.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset file to evaluate.",
    )
    parser.add_argument(
        "--check_grammar",
        dest="check_grammar",
        action="store_true",
        default=True,
        help="Run grammar checking using DistilBERT-CoLA (default: True).",
    )
    parser.add_argument(
        "--no_check_grammar",
        dest="check_grammar",
        action="store_false",
        help="Skip grammar checking.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Evaluate only a random sample of N lines (default: evaluate all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Process grammar evaluation in smaller batches to reduce memory usage (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON (default: print to console only).",
    )
    args = parser.parse_args()

    print(f"--- Dataset Evaluation for PocketNarrator ---")
    print(f"Dataset: {args.data}")
    print(f"Grammar Check: {args.check_grammar}")
    print(f"Sample Size: {args.sample_size if args.sample_size else 'Full dataset'}")
    print()

    # load dataset
    print("Loading dataset...")
    try:
        dataset_lines = load_text_dataset(args.data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loaded {len(dataset_lines)} lines from dataset.")

    # sample if requested
    if args.sample_size is not None and args.sample_size < len(dataset_lines):
        import random
        random.seed(42)
        dataset_lines = random.sample(dataset_lines, args.sample_size)
        print(f"Sampled {len(dataset_lines)} lines for evaluation.")

    # run evaluation
    print("\nRunning evaluation metrics...")
    
    if args.check_grammar and args.batch_size and args.batch_size < len(dataset_lines):
        print(f"Processing grammar evaluation in batches of {args.batch_size}...")
        
        # diversity metrics on full dataset
        from pocket_narrator.evaluate import distinct_n, repetition_rate
        evaluation_results = {
            "distinct_1": distinct_n(dataset_lines, n=1),
            "distinct_2": distinct_n(dataset_lines, n=2),
            "distinct_3": distinct_n(dataset_lines, n=3),
            "repetition_rate": repetition_rate(dataset_lines),
        }
        
        # grammar in batches
        from pocket_narrator.evaluate import calculate_grammar_score
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # Force CPU to avoid MPS memory issues
        else:
            device = "cpu"
        
        grammar_scores = []
        for i in range(0, len(dataset_lines), args.batch_size):
            batch = dataset_lines[i:i + args.batch_size]
            score = calculate_grammar_score(batch, device=device)
            if score > 0:
                grammar_scores.append(score)
            print(f"  Processed {min(i + args.batch_size, len(dataset_lines))}/{len(dataset_lines)} lines...")
        
        evaluation_results["grammar_score"] = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0.0
    else:
        evaluation_results = run_dataset_evaluation(
            dataset_text=dataset_lines,
            check_grammar=args.check_grammar,
        )

    # print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, value in evaluation_results.items():
        if value is not None:
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: N/A")
    print("="*60)

    # save results (optional)
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": args.data,
            "sample_size": args.sample_size if args.sample_size else len(dataset_lines),
            "total_lines": len(dataset_lines),
            "check_grammar": args.check_grammar,
            "results": evaluation_results,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
