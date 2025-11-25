"""
Model Comparison and Evaluation Script for PocketNarrator.

This script compares multiple models on validation data and curated prompts,
logging results to wandb with side-by-side comparisons, metrics tables, and
per-model summaries.

Usage:
    PYTHONPATH=. python3 scripts/evaluate.py \\
        --models models/ngram_20251125_185557.model models/transformer_20251125_220219.model \\
        --validation data/processed/TinyStories/TinyStoriesV2-GPT4-val.txt \\
        --run_name "Model Comparison Run 1"
        --prompts_file config/prompts.txt \\
        --max_new_tokens 50 \\
        --temperature 0.7 \\
        --wandb_project pocket-narrator-eval \\
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import math
import time
import torch
import wandb
from datetime import datetime
from collections import defaultdict

from pocket_narrator.models import load_model
from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.evaluate import (
    run_evaluation,
    calculate_perplexity,
    calculate_bleu,
    calculate_rouge_n,
    calculate_rouge_l,
    distinct_n,
    repetition_rate,
)
from pocket_narrator.data_loader import load_text_dataset, batchify_text
from pocket_narrator.trainers.transformer_trainer import TransformerTrainer
from pocket_narrator.trainers.ngram_trainer import NGramTrainer


# --- Default Prompts ---
DEFAULT_PROMPTS = [
    "Once upon a time,",
    "In a land far away,",
    "The queen ruled over her kingdom with",
    "A mysterious shadow appeared",
    "Technology has changed",
    "The future will be",
]


def load_prompts(prompts_path: str = None) -> list[str]:
    """Load prompts from file or use defaults."""
    if prompts_path and Path(prompts_path).exists():
        with open(prompts_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
        return prompts
    else:
        print(f"Using {len(DEFAULT_PROMPTS)} default prompts")
        return DEFAULT_PROMPTS


def infer_model_type_and_tokenizer(model_path: str) -> tuple[str, str]:
    """
    Attempt to infer model type and tokenizer type from model file.
    Returns (model_type, tokenizer_type) tuple.
    """
    model_path_str = str(model_path)
    
    # Try to load metadata from model if available
    if model_path_str.endswith('.json') or model_path_str.endswith('.model'):
        try:
            with open(model_path_str, 'r') as f:
                data = json.load(f)
            config = data.get('config', {})
            model_type = config.get('model_type', 'ngram')
            tokenizer_type = config.get('tokenizer_type', 'character')
            return model_type, tokenizer_type
        except:
            return 'ngram', 'character'
    elif model_path_str.endswith('.pth'):
        # Assume transformer
        return 'transformer', 'bpe'
    
    return 'ngram', 'character'


def get_tokenizer_for_model(model_type: str, tokenizer_type: str = None) -> object:
    """
    Load tokenizer matching the model type.
    Tries to infer from standard paths if not provided.
    """
    if tokenizer_type is None:
        tokenizer_type = 'bpe' if model_type == 'transformer' else 'character'
    
    tokenizer_path = f"tokenizers/{tokenizer_type}_tokenizer/"
    
    try:
        tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded {tokenizer_type} tokenizer from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        print(f"WARNING: Failed to load tokenizer from {tokenizer_path}: {e}")
        # Try fallback paths
        for alt_path in [
            "tokenizers/bpe_tokenizer/",
            "tokenizers/character_tokenizer/",
            "tokenizers/new_char_tokenizer/",
        ]:
            try:
                alt_type = 'bpe' if 'bpe' in alt_path else 'character'
                tokenizer = get_tokenizer(
                    tokenizer_type=alt_type,
                    tokenizer_path=alt_path,
                )
                print(f"Loaded {alt_type} tokenizer from {alt_path} (fallback)")
                return tokenizer
            except:
                continue
        raise ValueError(f"Could not load any tokenizer for model type {model_type}")


def compute_validation_metrics(model, tokenizer, val_lines: list[str], model_type: str, device: str) -> dict:
    """
    Compute validation loss and perplexity using appropriate trainer.
    """
    if not val_lines or len(val_lines) == 0:
        return {"val_loss": None, "val_perplexity": None}
    
    metrics = {}
    
    try:
        if model_type == "transformer":
            trainer = TransformerTrainer()
            val_loss = trainer.calculate_validation_loss(model, tokenizer, val_lines)
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = math.exp(val_loss) if val_loss != float('inf') else float('inf')
        elif model_type == "ngram":
            trainer = NGramTrainer()
            # NGram doesn't have built-in validation; just return None
            metrics["val_loss"] = None
            metrics["val_perplexity"] = None
        else:
            metrics["val_loss"] = None
            metrics["val_perplexity"] = None
    except Exception as e:
        print(f"WARNING: Could not compute validation metrics: {e}")
        metrics["val_loss"] = None
        metrics["val_perplexity"] = None
    
    return metrics


def generate_for_prompts(
    model,
    tokenizer,
    prompts: list[str],
    model_type: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Generate outputs for each prompt using the model.
    Returns list of (prompt, generated_text) tuples.
    """
    torch.manual_seed(seed)
    results = []
    
    for prompt in prompts:
        try:
            # Encode prompt
            if hasattr(tokenizer, 'encode'):
                prompt_tokens = tokenizer.encode(prompt)
            else:
                prompt_tokens = [tokenizer.token_to_id(t) for t in prompt.split()]
            
            # Create input batch
            input_batch = [prompt_tokens]
            
            # Generate using model's predict_sequence_batch
            sampling_cfg = {
                "strategy": "sample" if temperature > 0 else "greedy",
            }
            if temperature > 0:
                sampling_cfg["temperature"] = temperature
                if top_k > 0:
                    sampling_cfg["top_k"] = top_k
                if top_p < 1.0:
                    sampling_cfg["top_p"] = top_p
            
            output_tokens = model.predict_sequence_batch(input_batch, max_len=max_new_tokens, **sampling_cfg)
            
            # Decode output
            if hasattr(tokenizer, 'decode'):
                generated_text = tokenizer.decode(output_tokens[0])
            else:
                generated_text = " ".join([tokenizer.id_to_token(t) for t in output_tokens[0]])
            
            results.append((prompt, generated_text))
        except Exception as e:
            print(f"WARNING: Generation failed for prompt '{prompt}': {e}")
            results.append((prompt, f"[ERROR: {str(e)[:50]}]"))
    
    return results


def compute_generation_metrics(prompt: str, generated_text: str, target_text: str = None) -> dict:
    """
    Compute metrics for a generated output.
    """
    metrics = {}
    
    # Distinct n-grams
    for n in (1, 2, 3):
        metrics[f"distinct_{n}"] = distinct_n([generated_text], n=n)
    
    # Repetition
    metrics["repetition_rate"] = repetition_rate([generated_text])
    
    # If target text available, compute overlap metrics
    if target_text:
        metrics["bleu_4"] = calculate_bleu(generated_text, target_text)
        metrics["rouge_1"] = calculate_rouge_n(generated_text, target_text, n=1)
        metrics["rouge_2"] = calculate_rouge_n(generated_text, target_text, n=2)
        metrics["rouge_l"] = calculate_rouge_l(generated_text, target_text)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple PocketNarrator models on validation data and prompts."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to model files to compare (e.g., model1.model model2.pth)",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        help="Path to validation dataset file (one line per sample)",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to file with generation prompts (one per line). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy, >0 = sampling)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter (0 = disabled)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="pocket-narrator-eval",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Weights & Biases run name. If not provided, uses timestamp.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use. If not provided, auto-selects cuda > mps > cpu.",
    )
    
    args = parser.parse_args()
    
    # --- Auto-select device if not provided ---
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")
    
    # --- Load validation data if provided ---
    val_lines = None
    if args.validation:
        print(f"Loading validation data from {args.validation}...")
        val_lines = load_text_dataset(args.validation)
        print(f"Loaded {len(val_lines)} validation samples")
    
    # --- Load prompts ---
    prompts = load_prompts(args.prompts_file)
    
    # --- Initialize wandb ---
    run_name = args.run_name or f"model-eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "models": args.models,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
                "device": args.device,
            },
        )
    
    print(f"\n{'='*80}")
    print(f"Starting Model Evaluation: {run_name}")
    print(f"Models: {args.models}")
    print(f"{'='*80}\n")
    
    # --- Setup wandb table ---
    table_columns = [
        "model_name",
        "prompt",
        "generated_text",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_rate",
    ]
    eval_table = wandb.Table(columns=table_columns) if not args.disable_wandb else None
    
    model_summaries = {}
    
    # --- Evaluate each model ---
    for model_path in args.models:
        print(f"\n--- Evaluating {model_path} ---")
        start_time = time.time()
        
        # Infer model and tokenizer type
        model_type, tokenizer_type = infer_model_type_and_tokenizer(model_path)
        print(f"Inferred model_type={model_type}, tokenizer_type={tokenizer_type}")
        
        # Load model and tokenizer
        try:
            model = load_model(model_path)
            model.to(args.device)
            tokenizer = get_tokenizer_for_model(model_type, tokenizer_type)
        except Exception as e:
            print(f"ERROR: Failed to load model from {model_path}: {e}")
            continue
        
        model_name = Path(model_path).stem
        
        # Compute validation metrics
        print("Computing validation metrics...")
        val_metrics = compute_validation_metrics(model, tokenizer, val_lines, model_type, args.device)
        
        # Generate for prompts
        print(f"Generating for {len(prompts)} prompts...")
        generations = generate_for_prompts(
            model, tokenizer, prompts,
            model_type=model_type,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        
        # Compute generation metrics and log to table
        gen_metrics_all = []
        for prompt, generated_text in generations:
            gen_metrics = compute_generation_metrics(prompt, generated_text)
            gen_metrics_all.append(gen_metrics)
            
            if eval_table is not None:
                row = [
                    model_name,
                    prompt,
                    generated_text,
                    gen_metrics.get("distinct_1", 0),
                    gen_metrics.get("distinct_2", 0),
                    gen_metrics.get("distinct_3", 0),
                    gen_metrics.get("repetition_rate", 0),
                ]
                eval_table.add_data(*row)
        
        # Aggregate metrics
        elapsed_time = time.time() - start_time
        
        avg_gen_metrics = defaultdict(float)
        for metrics in gen_metrics_all:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    avg_gen_metrics[k] += v
        
        for k in avg_gen_metrics:
            avg_gen_metrics[k] /= len(gen_metrics_all) if gen_metrics_all else 1
        
        model_summaries[model_name] = {
            "model_path": model_path,
            "model_type": model_type,
            "tokenizer_type": tokenizer_type,
            "val_loss": val_metrics.get("val_loss"),
            "val_perplexity": val_metrics.get("val_perplexity"),
            "num_params": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0,
            "vocab_size": tokenizer.get_vocab_size(),
            "device": args.device,
            "eval_time_seconds": elapsed_time,
            **{f"avg_gen_{k}": v for k, v in avg_gen_metrics.items()},
        }
        
        # Print summary for this model
        print(f"\nModel: {model_name}")
        print(f"  Val Perplexity: {val_metrics.get('val_perplexity', 'N/A')}")
        print(f"  Avg Distinct-1: {avg_gen_metrics.get('distinct_1', 0):.4f}")
        print(f"  Avg Distinct-2: {avg_gen_metrics.get('distinct_2', 0):.4f}")
        print(f"  Avg Repetition: {avg_gen_metrics.get('repetition_rate', 0):.4f}")
        print(f"  Num Params: {model_summaries[model_name]['num_params']:,}")
        print(f"  Vocab Size: {model_summaries[model_name]['vocab_size']}")
        print(f"  Eval Time: {elapsed_time:.2f}s")
    
    # --- Log results to wandb ---
    if not args.disable_wandb:
        # Log evaluation table
        wandb.log({"evaluation_table": eval_table})
        
        # Log per-model summaries
        for model_name, summary in model_summaries.items():
            for metric_name, metric_value in summary.items():
                if isinstance(metric_value, (int, float)):
                    wandb.summary[f"{model_name}/{metric_name}"] = metric_value
        
        # Log detailed JSON summaries
        summary_json_path = f"eval_summary_{run_name}.json"
        with open(summary_json_path, 'w') as f:
            json.dump(model_summaries, f, indent=2)
        
        artifact = wandb.Artifact("evaluation_summary", type="evaluation")
        artifact.add_file(summary_json_path)
        wandb.log_artifact(artifact)
        
        # Log comparison results
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*80}")
        for model_name, summary in model_summaries.items():
            print(f"\n{model_name}:")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        print(f"\nResults logged to wandb project '{args.wandb_project}', run '{run_name}'")
        wandb.finish()
    else:
        print("\nWandb logging disabled. Results saved locally only.")
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*80}")
        for model_name, summary in model_summaries.items():
            print(f"\n{model_name}:")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
