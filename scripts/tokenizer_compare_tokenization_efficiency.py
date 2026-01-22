"""
Script to compare tokenization efficiency across multiple BPE tokenizers.

This script takes up to 5 trained BPE tokenizer paths and a dataset,
tokenizes the complete dataset with each tokenizer, and outputs
how many tokens each tokenizer needed.

Usage:
    PYTHONPATH=. python3 scripts/tokenizer_vocab_size_correction.py \
        --dataset data/processed/TinyStories/TinyStoriesV2-GPT4-train.bos_eos.txt \
        --tokenizers tokenizers/bpe_tokenizer_1k tokenizers/bpe_tokenizer_5k tokenizers/bpe_tokenizer_10k
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from tqdm import tqdm

from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.data_loader import load_text_dataset


def count_tokens_for_tokenizer(tokenizer, dataset_lines: list[str]) -> int:
    """
    Tokenize all lines in the dataset and return total token count.
    """
    total_tokens = 0
    for line in tqdm(dataset_lines, desc="Tokenizing", leave=False):
        tokens = tokenizer.encode(line)
        total_tokens += len(tokens)
    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Compare tokenization efficiency across multiple BPE tokenizers."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset file to tokenize.",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        required=True,
        help="Paths to up to 5 trained BPE tokenizer directories.",
    )
    args = parser.parse_args()

    # Validate number of tokenizers
    if len(args.tokenizers) > 5:
        print("WARNING: Only the first 5 tokenizers will be used.")
        args.tokenizers = args.tokenizers[:5]

    # Load the dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset_lines = load_text_dataset(args.dataset)
    print(f"Dataset loaded: {len(dataset_lines)} lines.")

    # Calculate total characters for reference
    total_chars = sum(len(line) for line in dataset_lines)
    print(f"Total characters in dataset: {total_chars:,}")

    print("\n" + "=" * 70)
    print("TOKENIZATION RESULTS")
    print("=" * 70)

    results = []

    for tokenizer_path in args.tokenizers:
        print(f"\nProcessing tokenizer: {tokenizer_path}")
        
        try:
            # Load the tokenizer
            tokenizer = get_tokenizer(
                tokenizer_type="bpe",
                tokenizer_path=tokenizer_path,
            )
            vocab_size = tokenizer.get_vocab_size()
            print(f"  Vocab size: {vocab_size:,}")

            # Count tokens
            total_tokens = count_tokens_for_tokenizer(tokenizer, dataset_lines)
            
            # Calculate compression ratio (chars per token)
            compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            
            results.append({
                "path": tokenizer_path,
                "vocab_size": vocab_size,
                "total_tokens": total_tokens,
                "compression_ratio": compression_ratio,
            })
            
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Compression ratio: {compression_ratio:.2f} chars/token")

        except FileNotFoundError as e:
            print(f"  ERROR: Tokenizer not found at {tokenizer_path}")
            print(f"  {e}")
        except Exception as e:
            print(f"  ERROR: Failed to process tokenizer: {e}")

    # Print summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Tokenizer Path':<40} {'Vocab Size':>12} {'Total Tokens':>15} {'Chars/Token':>12}")
        print("-" * 70)
        
        for r in results:
            path_short = r["path"][-38:] if len(r["path"]) > 38 else r["path"]
            print(f"{path_short:<40} {r['vocab_size']:>12,} {r['total_tokens']:>15,} {r['compression_ratio']:>12.2f}")
        
        print("-" * 70)
        print(f"Dataset: {args.dataset}")
        print(f"Total lines: {len(dataset_lines):,}")
        print(f"Total characters: {total_chars:,}")

    print("\n--- Tokenizer comparison finished. ---")


if __name__ == "__main__":
    main()
