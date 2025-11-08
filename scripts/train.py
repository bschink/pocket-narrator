"""
The main training script for the PocketNarrator project.

This script connects all modular components (data loader, tokenizer, model, evaluator)
into a functional end-to-end pipeline.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pocket_narrator.models import get_model
from pocket_narrator.tokenizers import get_tokenizer 
from pocket_narrator.evaluate import run_evaluation
from pocket_narrator.data_loader import load_text_dataset, split_text, batchify_text
from pocket_narrator.trainers import get_trainer

# --- Constants and Configuration ---
DATA_PATH = "data/mvp_dataset.txt"
TOKENIZER_TYPE = "character"
TOKENIZER_PATH = "tokenizers/character_tokenizer_vocab.json"
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
    print(f"--- Starting Training Run for {MODEL_TYPE} Model ---")

    # --- Data Loading and Preparation ---
    print(f"Loading and splitting dataset from {DATA_PATH}...")
    all_lines = load_text_dataset(DATA_PATH)
    train_lines, val_lines = split_text(all_lines, val_ratio=VAL_RATIO, seed=RANDOM_SEED)
    print(f"Dataset loaded: {len(train_lines)} training samples, {len(val_lines)} validation samples.")

    # --- Initialization ---
    print("Initializing tokenizer and model...")
    tokenizer = get_tokenizer(
        tokenizer_type=TOKENIZER_TYPE, 
        tokenizer_path=TOKENIZER_PATH, 
        train_corpus=train_lines
    )

    model_specific_config = MODEL_CONFIG.copy()
    model_specific_config['eos_token_id'] = tokenizer.char_to_idx['<eos>']
    model = get_model(
        model_type=MODEL_TYPE,
        vocab_size=tokenizer.get_vocab_size(),
        **model_specific_config
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

    predicted_tokens_batch = model.predict_sequence_batch(val_inputs)
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

    # --- Save Trained Model ---
    model.save(MODEL_SAVE_PATH)
    
    print(f"\n--- Training run finished successfully! Model saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    main()