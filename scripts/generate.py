"""
The main generation (inference) script for the PocketNarrator project.

This script loads a trained model and tokenizer, and uses them to generate
a story continuation from a user-provided prompt.

How to Use:
  - To run with default paths and prompt:
    python scripts/generate.py. ------> requires the existence of a trained model at models/ngram_model.json

  - To provide your own prompt and model:

    PYTHONPATH=. python3 scripts/generate.py \
    --prompt "the tree is" \
    --model_path models/n_gram/ngram_20251125_171941.model \
    --generation_strategy sample \
    --no_repeat_ngram_size 3 \
    --tokenizer_type bpe \
    --tokenizer_path tokenizers/bpe_tokenizer

  - To see all options:
    python scripts/generate.py --help
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
from pocket_narrator.models import load_model
from pocket_narrator.tokenizers import get_tokenizer

def main():
    # --- set up argument parser ---
    parser = argparse.ArgumentParser(description="Generate a story from a prompt using a trained PocketNarrator model.")
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="a girl went to the",
        help="The starting text (prompt) for the story generation."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ngram_model.json",
        help="The path to the saved model artifact."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the saved tokenizer. If not provided, defaults to tokenizer-type-specific path."
    )
    parser.add_argument(
        "--generation_strategy",
        type=str,
        choices=["greedy", "sample"],
        default="greedy",
        help="How to pick next tokens: 'greedy' or 'sample'.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="If set (e.g. 3), avoid repeating n-grams of this size.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["character", "bpe"],
        default="character",
        help="Type of tokenizer to use ('character' or 'bpe')."
    )

    args = parser.parse_args()

    print("--- Starting Generation Script ---")
    
    # --- load the trained model ---
    try:
        model = load_model(args.model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a valid model file. You can create one by running:")
        print("  python scripts/train.py")
        return

    # --- load the tokenizer ---
    try:
        tokenizer = get_tokenizer(
            tokenizer_type=args.tokenizer_type,
            tokenizer_path=args.tokenizer_path
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure a valid tokenizer exists. You can create one by running:")
        print("  python scripts/train.py")
        return
    
    # --- prepare users prompt ---
    prompt_text = args.prompt
    print(f"\nInput prompt: '{prompt_text}'")
    
    # --- inference pipeline ---
    prompt_tokens_batch = [tokenizer.encode(prompt_text)]
    predicted_tokens_batch = model.predict_sequence_batch(
        prompt_tokens_batch,
        strategy=args.generation_strategy,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    predicted_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    
    # --- display result ---
    generated_continuation = predicted_text_batch[0]
    full_story = prompt_text + generated_continuation
    
    print(f"Generated text: '{full_story}'")
    
    print("\n--- Generation finished. ---")

if __name__ == "__main__":
    main()