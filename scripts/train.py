"""
The main training script for the PocketNarrator project.

This script connects all modular components (data loader, tokenizer, model, evaluator)
into a functional end-to-end pipeline.


PYTHONPATH=. python3 scripts/train.py \
  --config configs/ngram_bpe_GPT4_clean_quartered.yaml 

PYTHONPATH=. python3 scripts/train.py \
  --config configs/transformer_bpe_GPT4_clean_quartered.yaml 

  
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
import yaml
import wandb
import torch

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

        if len(tokens) < 2:
            continue # skip too short sequences
        
        # split the token sequence in the middle
        split_point = len(tokens) // 2

        if split_point == 0:
            continue

        input_tokens_batch.append(tokens[:split_point])
        target_tokens_batch.append(tokens[split_point:])
    
    return input_tokens_batch, target_tokens_batch

def main():

    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Train an n-gram PocketNarrator model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file with experiment settings.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help=f"Path to training dataset (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="bpe",
        help="Type of tokenizer to use ('character', 'bpe').",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to save/load tokenizer. If not provided, defaults to tokenizer-type-specific path.",
    )
    parser.add_argument(
        "--tokenizer_config",
        type=json.loads,
        default='{"vocab_size": 5000, "special_tokens": ["<|endoftext|>", "<|pad|>"], "merges_per_round": 200}',
        help='JSON string for tokenizer config (e.g., \'{"vocab_size": 1024}\').',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ngram",
        help="Type of model to train ('ngram', 'transformer').",
    )
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
        help=(
            "Filename for the model (e.g. 'my_experiment.model'). "
            "If not given, a name based on model type and timestamp will be used."
        ),
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True, 
        help="Use KV-Caching for validation generation (default: True). Set --no-cache to disable."
    )
    parser.add_argument('--no-cache', dest='use_cache', action='store_false')
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    # ---- Local runtime config (start from defaults / CLI, then override with YAML) ----
    data_path = args.data
    val_ratio = VAL_RATIO
    random_seed = RANDOM_SEED
    batch_size = BATCH_SIZE

    tokenizer_type = args.tokenizer_type
    tokenizer_path = args.tokenizer_path
    bpe_tokenizer_config = args.tokenizer_config  # for BPE only
    char_special_tokens = None  # will be filled from YAML

    model_type = args.model_type
    model_config = MODEL_CONFIG.copy()
    trainer_type = TRAINER_TYPE

    generation_strategy = args.generation_strategy
    no_repeat_ngram_size = args.no_repeat_ngram_size

    model_dir = args.model_dir
    model_name = args.model_name

    # --- YAML override, if provided ---
    cfg = None
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # data
        data_path = cfg["data"]["path"]
        val_ratio = cfg["data"]["val_ratio"]
        random_seed = cfg["data"]["random_seed"]
        batch_size = cfg["data"]["batch_size"]

        # tokenizer
        tokenizer_type = cfg["tokenizer"]["type"]
        tokenizer_path = cfg["tokenizer"]["path"]
        
        # Read tokenizer_config from YAML (for BPE)
        if "tokenizer_config" in cfg["tokenizer"]:
            bpe_tokenizer_config = cfg["tokenizer"]["tokenizer_config"]
        
        # for character tokenizer we may have special tokens
        char_special_tokens = cfg["tokenizer"].get("special_tokens", None)

        # model
        model_type = cfg["model"]["type"]
        if model_type == "ngram":
            model_config = {"n": cfg["model"]["n"]}
        else:
            # Read transformer model config from YAML
            model_config = {
                "d_model": cfg["model"].get("d_model", 128),
                "n_layers": cfg["model"].get("n_layers", 2),
                "n_head": cfg["model"].get("n_head", 2),
                "max_len": cfg["model"].get("max_len", 128),
                "dropout": cfg["model"].get("dropout", 0.1),
                "pos_encoding_type": cfg["model"].get("pos_encoding_type", "sinusoidal"),
                "attention_type": cfg["model"].get("attention_type", "multi_head"),
                "activation_type": cfg["model"].get("activation_type", "gelu"),
            }

        trainer_type = cfg["trainer"]["type"]

        # generation
        generation_strategy = cfg["generation"]["strategy"]
        no_repeat_ngram_size = cfg["generation"]["no_repeat_ngram_size"]

        # saving
        model_dir = cfg["saving"]["model_dir"]
        model_name = cfg["saving"]["model_name"]

    # --- Initialize Weights & Biases (W&B) run ---
    run_name = None
    if args.config is not None and cfg is not None:
        run_name = cfg.get("run_name", None)

    if run_name is None:
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Base wandb config (static)
    wandb_config = {
        "run_name": run_name,
        "config_file": args.config,
        "seed": random_seed,

        # data
        "data/path": data_path,
        "data/val_ratio": val_ratio,
        "data/batch_size": batch_size,

        # tokenizer
        "tokenizer/type": tokenizer_type,
        "tokenizer/path": tokenizer_path,

        # model
        "model/type": model_type,

        # trainer
        "trainer/type": trainer_type,

        # generation
        "generation/strategy": generation_strategy,
        "generation/no_repeat_ngram": no_repeat_ngram_size,
        "generation/use_cache": getattr(args, "use_cache", False),

        # saving
        "saving/model_dir": model_dir,
        "saving/model_name": model_name,
    }

    wandb.init(
        entity="once-upon-a-prompt",
        project="pocket-narrator",
        name=run_name,
        config=wandb_config,
    )

    print(f"--- Starting Training Run for {model_type} Model ---")

    # --- Data Loading and Preparation ---
    print(f"Loading and splitting dataset from {data_path}...")
    all_lines = load_text_dataset(data_path)
    train_lines, val_lines = split_text(all_lines, val_ratio=val_ratio, seed=random_seed)
    print(f"Dataset loaded: {len(train_lines)} training samples, {len(val_lines)} validation samples.")

    # --- Initialization ---
    print("Initializing tokenizer and model...")

    if tokenizer_path is None:
        tokenizer_path = f"tokenizers/{tokenizer_type}_tokenizer/"

    # Build tokenizer kwargs based on tokenizer type
    if tokenizer_type == "bpe":
        tokenizer_kwargs = bpe_tokenizer_config.copy()
    elif tokenizer_type == "character":
        tokenizer_kwargs = {}
        if char_special_tokens is not None:
            tokenizer_kwargs["special_tokens"] = char_special_tokens
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer = get_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
        **tokenizer_kwargs,
    )

    if not os.path.exists(tokenizer_path):
        print(f"INFO: Tokenizer not found at {tokenizer_path}. Training tokenizer...")
        tokenizer.train(train_lines)
        tokenizer.save(tokenizer_path)
        print(f"INFO: Tokenizer trained and saved to {tokenizer_path}.")
    else:
        print("INFO: Using pre-existing/loaded tokenizer.")

    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        print("WARNING: No <|endoftext|> token found in tokenizer.")

    pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        print("WARNING: No <|pad|> token found in tokenizer. Padding may not work correctly.")

    # --- DEBUG: check vocab sizes and token ranges ---
    print("\nDEBUG: Tokenizer Diagnostics:")
    print(f"DEBUG: tokenizer vocab_size: {tokenizer.get_vocab_size()}")
    print(f"DEBUG: tokenizer type: {tokenizer_type}")
    print(f"DEBUG: pad_token_id: {pad_token_id}")
    if hasattr(tokenizer, 'special_tokens'):
        print(f"DEBUG: special_tokens: {tokenizer.special_tokens}")
    if hasattr(tokenizer, 'merges'):
        print(f"DEBUG: number of BPE merges: {len(tokenizer.merges)}")

    model_config = model_config.copy()
    model_config["eos_token_id"] = eos_token_id

    model = get_model(
        model_type=model_type,
        vocab_size=tokenizer.get_vocab_size(),
        **model_config,
    )
    
    # Create trainer with config from YAML (if transformer)
    if trainer_type == "transformer" and cfg is not None:
        trainer_config = cfg.get("trainer", {})
        trainer = get_trainer(
            trainer_type=trainer_type,
            learning_rate=float(trainer_config.get("learning_rate", 3e-4)),
            epochs=int(trainer_config.get("epochs", 1)),
            batch_size=batch_size,
            weight_decay=float(trainer_config.get("weight_decay", 0.1)),
            grad_clip=float(trainer_config.get("grad_clip", 1.0)),
            warmup_steps=int(trainer_config.get("warmup_steps", 0)),
            use_amp=bool(trainer_config.get("use_amp", False)),
            kv_caching_enabled=bool(trainer_config.get("kv_caching_enabled", False)),
            pad_token_id=pad_token_id,
        )
    elif trainer_type == "transformer":
        trainer = get_trainer(trainer_type=trainer_type, pad_token_id=pad_token_id)
    else:
        trainer = get_trainer(trainer_type=trainer_type)

    # --- Append the wandb.config with architecture/tokenizer/trainer metadata ---
    extra_cfg = {}

    # tokenizer info
    try:
        extra_cfg["tokenizer/path"] = tokenizer_path
        extra_cfg["tokenizer/vocab_size"] = tokenizer.get_vocab_size()
        extra_cfg["tokenizer/special_tokens"] = getattr(tokenizer, "special_token_names", None)
        extra_cfg["model/n"] = model_config.get("n", None)
        extra_cfg["trainer/lr_schedule"] = "cosine_with_warmup"
        extra_cfg["device"] = model.device



    except Exception:
        pass

    # model architecture (Transformer / Mamba / Ngram)
    if hasattr(model, "config") and isinstance(model.config, dict):
        for k, v in model.config.items():
            extra_cfg[f"model/{k}"] = v

    # number of parameters
    try:
        num_params = sum(p.numel() for p in model.parameters())
        extra_cfg["model/num_parameters"] = num_params
    except Exception:
        pass

    # trainer hyperparameters
    trainer_fields = [
        "learning_rate", "epochs", "batch_size",
        "weight_decay", "grad_clip", "warmup_steps",
        "use_amp", "device", "kv_caching_enabled"
    ]
    for attr in trainer_fields:
        if hasattr(trainer, attr):
            extra_cfg[f"trainer/{attr}"] = getattr(trainer, attr)
    
    # Add scheduler details (only for transformer trainer)
    if trainer_type == "transformer":
        extra_cfg["trainer/scheduler_type"] = "cosine_with_warmup"
        extra_cfg["trainer/scheduler_warmup_steps"] = trainer.warmup_steps

    # update wandb config
    wandb.config.update(extra_cfg, allow_val_change=True)


    # --- Training ---
    print("\n--- Starting Model Training ---")
    if trainer_type == "ngram":
        model = trainer.train(model=model, tokenizer=tokenizer, train_data=train_lines, batch_size=batch_size)
    else:
        model = trainer.train(model=model, tokenizer=tokenizer,
                            train_data=train_lines, val_data=val_lines, batch_size=batch_size)
    print("Model training complete.")

    # --- Validation ---
    print("\n--- Starting Validation ---")
    val_batch_iterator = batchify_text(val_lines, batch_size=batch_size, shuffle=False)
    val_batch_text = next(val_batch_iterator)
    val_inputs, target_tokens_batch = prepare_batch(val_batch_text, tokenizer)

    if trainer_type == "transformer":
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
        val_loss = trainer.calculate_validation_loss(model, tokenizer, val_lines, loss_fn)
    else:
        val_loss = None # NGram doesn't support this  

    predict_kwargs = {
        "strategy": generation_strategy,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if model_type == "transformer":
        predict_kwargs["use_cache"] = args.use_cache
        predict_kwargs["max_length"] = model_config.get("max_len", 128)  # Use model's max_len for generation
    else:
        predict_kwargs["max_length"] = 100  # For n-gram models

    predicted_tokens_batch = model.predict_sequence_batch(
        val_inputs,
        **predict_kwargs
    )

    predicted_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    target_text_batch = tokenizer.decode_batch(target_tokens_batch)

    # --- Log generation samples to W&B ---
    wandb.log({
        "samples/input": tokenizer.decode(val_inputs[0]),
        "samples/prediction": predicted_text_batch[0],
        "samples/ground_truth": target_text_batch[0],
    })

    print(f"Validation Input: '{tokenizer.decode(val_inputs[0])}'")
    print(f"Model Prediction: '{predicted_text_batch[0]}'")
    print(f"Ground Truth: '{target_text_batch[0]}'")

    evaluation_summary = run_evaluation(
        predicted_tokens=predicted_tokens_batch,
        target_tokens=target_tokens_batch,
        predicted_text=predicted_text_batch,
        target_text=target_text_batch,
        val_loss=val_loss,
        check_grammar=True
    )

    print("\n--- Evaluation Summary ---")
    for metric, value in evaluation_summary.items():
        if value is None:
            print(f"{metric}: None")
        else:
            print(f"{metric}: {value:.4f}")

    
    # --- Log evaluation metrics to W&B ---
    wandb_metrics = {f"eval/{metric}": value for metric, value in evaluation_summary.items()}
    wandb_metrics["data/train_size"] = len(train_lines)
    wandb_metrics["data/val_size"] = len(val_lines)

    wandb.log(wandb_metrics)

    # --- Determine model save path ---
    os.makedirs(model_dir, exist_ok=True)

    if model_name:
        model_filename = model_name
    else:
        # default: <model_type>_YYYYMMDD_HHMMSS.model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{timestamp}.model"

    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print(
            f"ERROR: Model file '{model_path}' already exists. "
            f"Choose a different --model_name or delete the existing file."
        )
        sys.exit(1)

    # --- Save Trained Model ---
    model.save(model_path)

    print(f"\n--- Training run finished successfully! Model saved to {model_path} ---")
    
    # --- Add tokenizer_type to model config ---
    # This allows evaluate.py to infer the correct tokenizer for each model
    if model_path.endswith('.json') or model_path.endswith('.model'):
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            model_data['config']['tokenizer_type'] = tokenizer_type
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2)
            print(f"INFO: Added tokenizer_type='{tokenizer_type}' to model config")
        except Exception as e:
            print(f"WARNING: Could not update model config with tokenizer_type: {e}")
    elif model_path.endswith('.pth') or model_path.endswith('.model'):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            checkpoint['config']['tokenizer_type'] = tokenizer_type
            torch.save(checkpoint, model_path)
            print(f"INFO: Added tokenizer_type='{tokenizer_type}' to model config")
        except Exception as e:
            print(f"WARNING: Could not update model config with tokenizer_type: {e}")

    # --- Final W&B summary entries ---
    wandb.summary["model_filename"] = model_filename
    wandb.summary["model_path"] = model_path
    wandb.summary["train/num_samples"] = len(train_lines)
    wandb.summary["val/num_samples"] = len(val_lines)
    wandb.summary["finished_at"] = datetime.now().isoformat(timespec="seconds")

    # --- Log model artifacts ---
    wandb.save(model_path)

    # optionally save tokenizer directory as artifact
    try:
        tokenizer_artifact = wandb.Artifact(
            name=f"tokenizer-{run_name}",
            type="tokenizer",
            metadata={"tokenizer_type": tokenizer_type}
        )
        tokenizer_artifact.add_dir(tokenizer_path)
        wandb.run.log_artifact(tokenizer_artifact)
    except Exception:
        pass



    # --- Append metadata to model log to keep track ---
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "model_name": model_filename,
        "model_type": model_type,
        "trainer_type": trainer_type,
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": tokenizer_path,
        "data_path": data_path,
        "generation_strategy": generation_strategy,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "model_config": model_config,
        "val_ratio": val_ratio,
        "batch_size": batch_size,
    }

    log_path = os.path.join(model_dir, "model_log.json")

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

    # --- Finish W&B run ---
    wandb.finish()

if __name__ == "__main__":
    main()