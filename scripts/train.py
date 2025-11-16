"""
The main training script for the PocketNarrator project.

This script connects all modular components (data loader, tokenizer, model, evaluator)
into a functional end-to-end pipeline.



PYTHONPATH=. python3 scripts/train.py \
  --generation_strategy sample \
  --tokenizer_type character \
  --tokenizer_path tokenizers/character_tokenizer_vocab.json \
  --no_repeat_ngram_size 3 \
  --data data/processed/TinyStories/TinyStories-train.bos_eos_30k.txt

  
  PYTHONPATH=. python3 scripts/train.py \
  --data data/processed/TinyStories/TinyStoriesV2-GPT4-train.bos_eos.txt \
  --generation_strategy sample \
  --no_repeat_ngram_size 3 \
  --model_dir models/cool_models \
  --model_name new_ngram_allgpt4_tinystories.model \
  --tokenizer_type character \
  --tokenizer_path tokenizers/new_char_tokenizer 


"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
from datetime import datetime
import json

from pocket_narrator.models import get_model
from pocket_narrator.tokenizers import get_tokenizer 
from pocket_narrator.evaluate import run_evaluation
from pocket_narrator.data_loader import load_text_dataset, split_text, batchify_text
from pocket_narrator.trainers import get_trainer

# --- Constants and Configuration (which can get overridden by input arguments)---
DATA_PATH = "data/mvp_dataset.txt"
TOKENIZER_TYPE = "bpe"
TOKENIZER_PATH = "tokenizers/bpe_tokenizer/"
MODEL_TYPE = "ngram"
MODEL_CONFIG = {"n": 3}
MODEL_SAVE_PATH = "models/ngram_model.json"
TRAINER_TYPE = "ngram"
BATCH_SIZE = 2
VAL_RATIO = 0.2
RANDOM_SEED = 42


def prepare_batch(batch_text: list[str], tokenizer) -> tuple[list[list[int]], list[list[int]]]:
    """
    Helper function to prepare a batch of text for the model.
    Task: Given the first half of a sentence, predict the second half.
    """
    input_tokens_batch = []
    target_tokens_batch = []

    for text in batch_text:
        tokens = tokenizer.encode(text)
        # split the token sequence in the middle
        split_point = len(tokens) // 2
        input_tokens_batch.append(tokens[:split_point])
        target_tokens_batch.append(tokens[split_point:])
    
    return input_tokens_batch, target_tokens_batch

def main():

    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Train an n-gram PocketNarrator model.")
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help=f"Path to training dataset (default: {DATA_PATH})",
    )
    parser.add_argument("--tokenizer_type", type=str, default="bpe", help="Type of tokenizer to use ('character', 'bpe').")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to save/load tokenizer. If not provided, defaults to tokenizer-type-specific path.")
    parser.add_argument("--tokenizer_config", type=json.loads, default='{"vocab_size": 1024, "special_tokens": {"<bos>": 1025, "<eos>": 1026}}', help='JSON string for tokenizer config (e.g., \'{"vocab_size": 1024}\').')
    parser.add_argument(
        "--generation_strategy",
        type=str,
        choices=["greedy", "sample"],
        default="greedy",
        help="Generation strategy for validation output (does not affect training).",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="If set (e.g. 3), avoid repeating n-grams of this size during validation generation.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Filename for the model (e.g. 'my_experiment.model'). "
             "If not given, a name based on model type and timestamp will be used.",
    )
    args = parser.parse_args()
    
    print(f"--- Starting Training Run for {MODEL_TYPE} Model ---")

    # --- Data Loading and Preparation ---
    print(f"Loading and splitting dataset from {args.data}...")
    all_lines = load_text_dataset(args.data)
    train_lines, val_lines = split_text(all_lines, val_ratio=VAL_RATIO, seed=RANDOM_SEED)
    print(f"Dataset loaded: {len(train_lines)} training samples, {len(val_lines)} validation samples.")

    # --- Initialization ---
    print("Initializing tokenizer and model...")
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = f"tokenizers/{args.tokenizer_type}_tokenizer/"
    tokenizer = get_tokenizer(
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=tokenizer_path,
        **args.tokenizer_config
    )

    is_untrained = tokenizer.get_vocab_size() <= (256 + len(args.tokenizer_config.get("special_tokens", {})))
    if is_untrained and not os.path.exists(tokenizer_path):
        print(f"INFO: Tokenizer is untrained. Preparing training data...")
        if args.tokenizer_type == "bpe":
            # bpe tokenizer needs an iterator
            corpus_data = batchify_text(train_lines, batch_size=1000, shuffle=False)
        else: # assumes CharacterTokenizer or similar
            # can take the full list.
            corpus_data = train_lines
            
        tokenizer.train(corpus_data)
        tokenizer.save(tokenizer_path)
    else:
        print("INFO: Using pre-existing/loaded tokenizer.")

    model_config = MODEL_CONFIG.copy()
    
    # Get EOS token ID based on tokenizer type
    if hasattr(tokenizer, 'special_tokens') and '<eos>' in tokenizer.special_tokens:
        model_config['eos_token_id'] = tokenizer.special_tokens['<eos>']
    else:
        print("WARNING: No <eos> token found in tokenizer.")
        model_config['eos_token_id'] = None
    
    model = get_model(
        model_type=MODEL_TYPE,
        vocab_size=tokenizer.get_vocab_size(),
        **model_config
    )
    trainer = get_trainer(trainer_type=TRAINER_TYPE)

    # --- Training ---
    print("\n--- Starting Model Training ---")
    model = trainer.train(model=model, tokenizer=tokenizer, train_data=train_lines)
    print("Model training complete.")

    # --- Validation ---
    print("\n--- Starting Validation ---")
    val_batch_iterator = batchify_text(val_lines, batch_size=BATCH_SIZE, shuffle=False)
    val_batch_text = next(val_batch_iterator)
    val_inputs, target_tokens_batch = prepare_batch(val_batch_text, tokenizer)
    
    val_inputs, target_tokens_batch = prepare_batch(val_batch_text, tokenizer)

    predicted_tokens_batch = model.predict_sequence_batch(
        val_inputs,
        strategy=args.generation_strategy,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    predicted_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    target_text_batch = tokenizer.decode_batch(target_tokens_batch)

    print(f"Validation Input: '{tokenizer.decode(val_inputs[0])}'")
    print(f"Model Prediction: '{predicted_text_batch[0]}'")
    print(f"Ground Truth: '{target_text_batch[0]}'")

    evaluation_summary = run_evaluation(
        predicted_tokens=predicted_tokens_batch,
        target_tokens=target_tokens_batch,
        predicted_text=predicted_text_batch,
        target_text=target_text_batch
    )
    
    print("\n--- Evaluation Summary ---")
    for metric, value in evaluation_summary.items():
        print(f"{metric}: {value:.4f}")

        # --- Determine model save path ---
    os.makedirs(args.model_dir, exist_ok=True)

    if args.model_name:
        model_filename = args.model_name
    else:
        # default: <model_type>_YYYYMMDD_HHMMSS.model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{MODEL_TYPE}_{timestamp}.model"

    model_path = os.path.join(args.model_dir, model_filename)

    if os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' already exists. "
              f"Choose a different --model_name or delete the existing file.")
        sys.exit(1)


    # --- Save Trained Model ---
    model.save(model_path)
    
    print(f"\n--- Training run finished successfully! Model saved to {model_path} ---")

    # --- Append metadata to model log to keep track ---
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "model_name": model_filename,
        "model_type": MODEL_TYPE,
        "trainer_type": TRAINER_TYPE,
        "tokenizer_type": TOKENIZER_TYPE,
        "tokenizer_path": TOKENIZER_PATH,
        "data_path": args.data,
        "generation_strategy": args.generation_strategy,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "model_config": model_config,
        "val_ratio": VAL_RATIO,
        "batch_size": BATCH_SIZE,
    }

    log_path = os.path.join(args.model_dir, "model_log.json")
    
    # if file exists, load it; otherwise start a new list
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print("WARNING: model_log.json is malformed, creating a new one.")
            log_data = []
    else:
        log_data = []

    # append new entry
    log_data.append(log_entry)

    # write back pretty-printed JSON
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"Model metadata appended to log: {log_path}")


if __name__ == "__main__":
    main()