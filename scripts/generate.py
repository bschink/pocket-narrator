"""
The main generation (inference) script for the PocketNarrator project.

This script loads a trained model and uses it to generate a story continuation
from a user-provided prompt.

How to Use:
  - To run with the default prompt:
    python scripts/generate.py

  - To provide your own prompt:
    python scripts/generate.py --prompt "A girl went to the"

  - To see all options:
    python scripts/generate.py --help
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from pocket_narrator.model import load_model
from pocket_narrator.tokenizer import get_tokenizer

def main():
    # --- set up argument parser ---
    parser = argparse.ArgumentParser(description="Generate a story from a prompt using a trained PocketNarrator model.")
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time there was",
        help="The starting text (prompt) for the story generation."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/mvp_model.pth",
        help="The path to the saved model artifact."
    )
    args = parser.parse_args()

    print("--- Starting Generation Script ---")
    
    # --- load the trained model ---
    try:
        model = load_model(args.model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure you have a valid model file. You can create one by running:")
        print("  python scripts/train.py")
        return

    # --- initialize other components ---
    tokenizer = get_tokenizer()
    
    # --- prepare the user's prompt ---
    prompt_text = args.prompt
    print(f"Input prompt: '{prompt_text}'")
    
    # --- inference pipeline ---
    prompt_tokens_batch = [tokenizer.encode(prompt_text)]
    predicted_tokens_batch = model.predict_sequence_batch(prompt_tokens_batch)
    predicted_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    
    generated_text = predicted_text_batch[0]
    print(f"Generated text: '{generated_text}'")
    
    print("--- Generation finished ---")

if __name__ == "__main__":
    main()