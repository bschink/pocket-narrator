"""
The main training script for the PocketNarrator project.

This script connects all modular components (data loader, tokenizer, model, evaluator)
into a functional end-to-end pipeline.
"""
import sys
from pathlib import Path

# Add the project root to Python path for robust imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pocket_narrator.model import get_model
from pocket_narrator.tokenizer import get_tokenizer
from pocket_narrator.evaluate import run_evaluation
from pocket_narrator.data_loader import load_text_dataset, split_text, batchify_text

# --- Constants and Configuration ---
DATA_PATH = "data/mvp_dataset.txt"
MODEL_SAVE_PATH = "models/mvp_model.pth"
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
    print("--- Starting MVP for PocketNarrator ---")

    # --- Initialization ---
    print("Initializing tokenizer and model...")
    tokenizer = get_tokenizer(tokenizer_type="simple")
    model = get_model(model_type="mvp", vocab_size=tokenizer.get_vocab_size())

    # --- Data Loading and Preparation ---
    print(f"Loading and splitting dataset from {DATA_PATH}...")
    all_lines = load_text_dataset(DATA_PATH)
    train_lines, val_lines = split_text(all_lines, val_ratio=VAL_RATIO, seed=RANDOM_SEED)
    print(f"Dataset loaded: {len(train_lines)} training samples, {len(val_lines)} validation samples.")

    # --- MVP Training Loop (Simulated) ---
    print("\n--- Starting MVP Training Loop (simulating one step) ---")
    train_batch_iterator = batchify_text(train_lines, batch_size=BATCH_SIZE, shuffle=True, seed=RANDOM_SEED)
    train_batch_text = next(train_batch_iterator)
    train_inputs, train_targets = prepare_batch(train_batch_text, tokenizer)
    
    # use the model to get a prediction. later we would compute loss and backpropagate
    _ = model.predict_sequence_batch(train_inputs)
    print(f"Processed one training batch of size {len(train_batch_text)}.")
    print(f"  - Example Input (tokens): {train_inputs[0]}")
    print(f"  - Example Target (tokens): {train_targets[0]}")

    # --- MVP Validation Loop (Simulated) ---
    print("\n--- Starting MVP Validation ---")
    val_batch_iterator = batchify_text(val_lines, batch_size=BATCH_SIZE, shuffle=False)
    val_batch_text = next(val_batch_iterator)
    val_inputs, target_tokens_batch = prepare_batch(val_batch_text, tokenizer)

    # use the model to get a prediction
    predicted_tokens_batch = model.predict_sequence_batch(val_inputs)
    
    # decode text representations for evaluation
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

    # --- Save the Trained Model ---
    model.save(MODEL_SAVE_PATH)
    
    print("\n--- MVP run finished successfully! ---")


if __name__ == "__main__":
    main()