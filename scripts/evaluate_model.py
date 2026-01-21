"""
Model Evaluation Script

Evaluates a trained language model by:
1. Loading the model (ngram, transformer, or mamba)
2. Loading a dataset (test, training, or validation)
3. Splitting stories at the midpoint (prompt | ground_truth)
4. Generating predictions from prompts using the model
5. Evaluating predictions with all available metrics

Metrics evaluated:
  - distinct_n (1, 2, 3) - on generated predictions
  - repetition_rate - on generated predictions
  - grammar_score - on generated predictions
  - llm_judge - on generated predictions
  - text_quality (coherence, cohesion) - on generated predictions
  - noun_carryover - on generated predictions
  - word_count, sentence_count - on generated predictions
  - BLEU - between generated and ground truth
  - ROUGE-1, ROUGE-2, ROUGE-L - between generated and ground truth
  - perplexity - model's likelihood of ground truth

Logs to W&B with:
  - Results table with all metrics per story
  - Summary statistics (mean, min, max)
  - Category plots
  - JSON artifact with full results

Usage:
    python scripts/evaluate_model.py \
        --model_path models/transformer/transformer_model.pth \
        --model_type transformer \
        --dataset_path data/test_dataset.txt \
        --output_path results/model_eval.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys
import yaml
import random

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

from pocket_narrator.models import load_model
from pocket_narrator.tokenizers import get_tokenizer
from pocket_narrator.evaluate import (
    distinct_n,
    repetition_rate,
    calculate_grammar_score,
    count_words,
    count_sentences,
    run_llm_judge_evaluation,
    calculate_bleu, #unique to model eval
    calculate_rouge_n, #unique to model eval
    calculate_rouge_l, #unique to model eval
    _HAS_TEXT_QUALITY,
    _HAS_NOUN_CARRYOVER,
)

if _HAS_TEXT_QUALITY:
    from pocket_narrator.text_quality import TextQualityConfig, evaluate_text_quality, _Embedder

if _HAS_NOUN_CARRYOVER:
    from pocket_narrator.noun_carryover import noun_carryover_metrics, SoftConfig, SoftEmbedder


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def split_story_at_midpoint(story: str) -> Tuple[str, str]:
    """
    Split a story at the midpoint (50%), matching the training script behavior.
    
    This mirrors the prepare_batch() function in scripts/train.py:
    - Split token sequence in the middle
    - Used for "given the first half, predict the second half" task
    
    Args:
        story: Full story text
        
    Returns:
        (prompt, ground_truth) tuple
    """
    story = story.strip()
    if not story:
        return "", ""
    
    # Split at character midpoint
    mid = len(story) // 2
    return story[:mid].strip(), story[mid:].strip()


def load_dataset(dataset_path: str) -> List[str]:
    """Load dataset from text file (one story per line or separated by newlines)."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        stories = [line.strip() for line in f if line.strip()]
    
    return stories


def generate_from_prompt(
    model,
    tokenizer,
    prompt: str,
    model_type: str,
    generation_kwargs: Dict = None
) -> str:
    """
    Generate a continuation from a prompt using the model.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        prompt: Text prompt to generate from
        model_type: Type of model ('ngram', 'transformer', 'mamba')
        generation_kwargs: Additional generation parameters
        
    Returns:
        Generated text continuation
    """
    if generation_kwargs is None:
        generation_kwargs = {}
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens_batch = [prompt_tokens]
    
    # Build generation parameters based on model type
    predict_kwargs = {
        "strategy": generation_kwargs.get("strategy", "greedy"),
        "max_length": generation_kwargs.get("max_length", 200),
    }
    
    if model_type == "ngram":
        if "no_repeat_ngram_size" in generation_kwargs:
            predict_kwargs["no_repeat_ngram_size"] = generation_kwargs["no_repeat_ngram_size"]
    elif model_type == "transformer":
        predict_kwargs["temperature"] = generation_kwargs.get("temperature", 1.0)
        if "top_k" in generation_kwargs:
            predict_kwargs["top_k"] = generation_kwargs["top_k"]
        if "top_p" in generation_kwargs:
            predict_kwargs["top_p"] = generation_kwargs["top_p"]
    
    # Generate
    predicted_tokens_batch = model.predict_sequence_batch(
        prompt_tokens_batch,
        **predict_kwargs
    )
    
    # Decode
    generated_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    
    return generated_text_batch[0]


def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    model_type: str,
    device: str = "cpu"
) -> float:
    """
    Calculate perplexity of a text under the model.
    
    Perplexity = exp(cross_entropy_loss)
    
    Uses forward pass + cross-entropy loss for neural models.
    Uses probability-based loss for n-gram models.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        text: Text to compute perplexity for
        model_type: Type of model
        device: Device to use
        
    Returns:
        Perplexity score (lower is better)
    """
    try:
        tokens = tokenizer.encode(text)
        
        if model_type == "ngram":
            # For n-gram models, compute cross-entropy loss from probabilities
            loss = 0.0
            for i in range(1, len(tokens)):
                context = tokens[:i]
                next_token = tokens[i]
                
                try:
                    if hasattr(model, 'get_token_probability'):
                        prob = model.get_token_probability(next_token, context)
                    else:
                        prob = 1.0 / model.vocab_size if hasattr(model, 'vocab_size') else 1e-4
                except:
                    prob = 1e-4
                
                loss += -math.log(max(prob, 1e-10))
            
            avg_loss = loss / max(len(tokens), 1)
            return math.exp(avg_loss)
        
        elif model_type in ["transformer", "mamba"]:
            # For neural models: forward pass + cross-entropy loss
            import torch
            import torch.nn.functional as F
            
            if len(tokens) < 2:
                return float('nan')
            
            try:
                # Prepare input and target for language modeling
                # Input: all tokens except last, Target: all tokens except first
                input_ids = torch.tensor([tokens[:-1]], device=device)
                target_ids = torch.tensor([tokens[1:]], device=device)
                
                with torch.no_grad():
                    # Forward pass to get logits
                    output = model(input_ids)
                    
                    # Handle different output formats (could be tuple or tensor)
                    if isinstance(output, tuple):
                        logits = output[0]  # Usually (logits, hidden_state, ...)
                    else:
                        logits = output
                    
                    # Compute cross-entropy loss
                    # Reshape: (batch * seq_len, vocab_size) vs (batch * seq_len,)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1),
                        reduction='mean'
                    )
                    
                    return math.exp(loss.item())
            except Exception as e:
                print(f"Warning: Forward pass perplexity computation failed: {e}")
                return float('nan')
        
        else:
            return float('nan')
            
    except Exception as e:
        print(f"Warning: Could not compute perplexity: {e}")
        return float('nan')


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_story(
    prompt: str,
    generated: str,
    ground_truth: str,
    model=None,
    tokenizer=None,
    model_type: str = "transformer",
    device: str = "cpu",
    metrics_config: Optional[Dict] = None,
    text_quality_config: Optional['TextQualityConfig'] = None,
    text_quality_embedder: Optional['_Embedder'] = None,
    noun_carryover_config: Optional['SoftConfig'] = None,
    noun_carryover_embedder: Optional['SoftEmbedder'] = None,
) -> Dict:
    """
    Evaluate a single prompt-generated-ground_truth triple.
    
    Args:
        prompt: The prompt given to the model
        generated: The model's generated text
        ground_truth: The reference ground truth text
        model: The loaded model (for perplexity)
        tokenizer: The loaded tokenizer (for perplexity)
        model_type: Type of model
        device: Device to use
        text_quality_config: Configuration for text quality metrics
        text_quality_embedder: Embedder for text quality
        noun_carryover_config: Configuration for noun carryover metrics
        noun_carryover_embedder: Embedder for noun carryover
        
    Returns:
        Dictionary with metric results
    """
    results = {
        "prompt": prompt,
        "generated": generated,
        "ground_truth": ground_truth,
    }
    
    # --- 1. Diversity Metrics (on generated text)
    try:
        results["distinct_1"] = distinct_n([generated], n=1)
        results["distinct_2"] = distinct_n([generated], n=2)
        results["distinct_3"] = distinct_n([generated], n=3)
    except Exception as e:
        print(f"Warning: Error computing distinct_n: {e}")
        results["distinct_1"] = None
        results["distinct_2"] = None
        results["distinct_3"] = None
    
    # --- 2. Repetition Rate (on generated text)
    try:
        results["repetition_rate"] = repetition_rate([generated])
    except Exception as e:
        print(f"Warning: Error computing repetition_rate: {e}")
        results["repetition_rate"] = None
    
    # --- 3. Grammar Score (on generated text)
    try:
        grammar_score = calculate_grammar_score([generated], device=device)
        results["grammar_score"] = grammar_score
    except Exception as e:
        print(f"Warning: Error computing grammar_score: {e}")
        results["grammar_score"] = None
    
    # --- 4. Statistics (on full story: prompt + generated)
    try:
        full_generated = prompt + generated
        results["word_count"] = count_words([full_generated])
        results["sentence_count"] = count_sentences([full_generated])
    except Exception as e:
        print(f"Warning: Error computing statistics: {e}")
        results["word_count"] = None
        results["sentence_count"] = None
    
    # --- 5. LLM Judge (on generated text)
    llm_judge_enabled = metrics_config.get("llm_judge", {}).get("enabled", False) if metrics_config else False
    if llm_judge_enabled:
        try:
            llm_results = run_llm_judge_evaluation([prompt], [generated])
            if llm_results:
                results["llm_judge_grammar"] = llm_results.get("llm_judge_grammar")
                results["llm_judge_creativity"] = llm_results.get("llm_judge_creativity")
                results["llm_judge_consistency"] = llm_results.get("llm_judge_consistency")
                results["llm_judge_num_evaluated"] = llm_results.get("llm_judge_num_evaluated")
                results["llm_judge_num_failed"] = llm_results.get("llm_judge_num_failed")
            else:
                results["llm_judge_grammar"] = None
                results["llm_judge_creativity"] = None
                results["llm_judge_consistency"] = None
                results["llm_judge_num_evaluated"] = None
                results["llm_judge_num_failed"] = None
        except Exception as e:
            print(f"Warning: Error computing llm_judge: {e}")
            results["llm_judge_grammar"] = None
            results["llm_judge_creativity"] = None
            results["llm_judge_consistency"] = None
            results["llm_judge_num_evaluated"] = None
            results["llm_judge_num_failed"] = None
    else:
        results["llm_judge_grammar"] = None
        results["llm_judge_creativity"] = None
        results["llm_judge_consistency"] = None
        results["llm_judge_num_evaluated"] = None
        results["llm_judge_num_failed"] = None
    
    # --- 6. Text Quality (on generated text)
    if _HAS_TEXT_QUALITY:
        try:
            tq_results = evaluate_text_quality(
                generated,
                cfg=text_quality_config,
                embedder=text_quality_embedder
            )
            results["text_quality_coherence"] = tq_results.get("coherence")
            results["text_quality_cohesion"] = tq_results.get("cohesion_mean")
            results["text_quality_score"] = tq_results.get("text_quality")
        except Exception as e:
            print(f"Warning: Error computing text_quality: {e}")
            results["text_quality_coherence"] = None
            results["text_quality_cohesion"] = None
            results["text_quality_score"] = None
    else:
        results["text_quality_coherence"] = None
        results["text_quality_cohesion"] = None
        results["text_quality_score"] = None
    
    # --- 7. Noun Carryover (on generated text)
    noun_carryover_enabled = metrics_config.get("noun_carryover", {}).get("enabled", True) if metrics_config else True
    if noun_carryover_enabled and _HAS_NOUN_CARRYOVER and prompt:
        try:
            soft_cfg = noun_carryover_config if noun_carryover_config else SoftConfig()
            nc_results = noun_carryover_metrics(prompt, generated, soft_cfg=soft_cfg)
            results["noun_carryover_hard_coverage"] = nc_results.get("hard_coverage")
            results["noun_carryover_hard_jaccard"] = nc_results.get("hard_jaccard")
            results["noun_carryover_hard_precision"] = nc_results.get("hard_precision")
            results["noun_carryover_soft_coverage"] = nc_results.get("soft_coverage")
            results[f"noun_carryover_soft_coverage@{soft_cfg.threshold:.2f}"] = nc_results.get(
                f"soft_coverage@{soft_cfg.threshold:.2f}"
            )
        except Exception as e:
            print(f"Warning: Error computing noun_carryover: {e}")
            results["noun_carryover_hard_coverage"] = None
            results["noun_carryover_hard_jaccard"] = None
            results["noun_carryover_hard_precision"] = None
            results["noun_carryover_soft_coverage"] = None
            results[f"noun_carryover_soft_coverage@0.70"] = None
    else:
        results["noun_carryover_hard_coverage"] = None
        results["noun_carryover_hard_jaccard"] = None
        results["noun_carryover_hard_precision"] = None
        results["noun_carryover_soft_coverage"] = None
        results[f"noun_carryover_soft_coverage@0.70"] = None
    
    # --- 8. N-gram Overlap Metrics (BLEU, ROUGE - generated vs ground_truth)
    try:
        results["bleu"] = calculate_bleu(generated, ground_truth)
    except Exception as e:
        print(f"Warning: Error computing BLEU: {e}")
        results["bleu"] = None
    
    try:
        results["rouge_1"] = calculate_rouge_n(generated, ground_truth, n=1)
        results["rouge_2"] = calculate_rouge_n(generated, ground_truth, n=2)
        results["rouge_l"] = calculate_rouge_l(generated, ground_truth)
    except Exception as e:
        print(f"Warning: Error computing ROUGE: {e}")
        results["rouge_1"] = None
        results["rouge_2"] = None
        results["rouge_l"] = None
    
    # --- 9. Perplexity (model's likelihood of ground truth)
    try:
        results["perplexity"] = calculate_perplexity(
            model,
            tokenizer,
            ground_truth,
            model_type,
            device=device
        )
    except Exception as e:
        print(f"Warning: Error computing perplexity: {e}")
        results["perplexity"] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained language model on a dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ngram", "transformer", "mamba"],
        default=None,
        help="Type of model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of dataset for logging"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["character", "bpe"],
        default=None,
        help="Type of tokenizer"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--max_stories",
        type=int,
        default=None,
        help="Maximum number of stories to evaluate"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (cuda, mps, or cpu). If not provided, auto-selects based on availability."
    )
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        print(f"Loaded config from {args.config}\n")
    
    # Extract values from config with CLI overrides (CLI > config > defaults)
    model_path = args.model_path or config.get("model", {}).get("path")
    model_type = args.model_type or config.get("model", {}).get("type", "transformer")
    dataset_path = args.dataset_path or config.get("dataset", {}).get("path")
    dataset_name = args.dataset_name or config.get("dataset", {}).get("name", "Dataset")
    max_stories = args.max_stories or config.get("dataset", {}).get("max_stories")
    tokenizer_type = args.tokenizer_type or config.get("tokenizer", {}).get("type", "character")
    tokenizer_path = args.tokenizer_path or config.get("tokenizer", {}).get("path")
    device = args.device or config.get("device", {}).get("type", "cpu")
    output_path = args.output_path or config.get("output", {}).get("json_path", "results/model_eval.json")
    wandb_project = args.wandb_project or config.get("output", {}).get("wandb_project", "pocket-narrator-model-eval")
    
    # Metrics from config (CLI args take precedence)
    metrics_config = config.get("metrics", {})
    
    # LLM Judge sampling settings
    llm_judge_config = metrics_config.get("llm_judge", {})
    llm_judge_sample_size = llm_judge_config.get("sample_size", 1000)
    llm_judge_random_seed = llm_judge_config.get("random_seed", 42)
    
    # Validate that required arguments are provided
    if not model_path:
        parser.error("--model_path or --config with model.path is required")
    if not dataset_path:
        parser.error("--dataset_path or --config with dataset.path is required")
    
    # Device detection if not explicitly set
    import torch
    if not device or device == "cpu":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Generation parameters
    generation_kwargs = config.get("generation", {})
    
    # --- Print Configuration ---
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Model Type: {model_type}")
    print(f"Dataset: {dataset_path}")
    print(f"Dataset Name: {dataset_name}")
    print(f"Max Stories: {max_stories or 'All'}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print(f"W&B Project: {wandb_project}")
    print("="*80 + "\n")
    
    # --- Load Model & Tokenizer ---
    print("Loading model and tokenizer...")
    try:
        model = load_model(model_path)
        model.to(device) if hasattr(model, 'to') else None
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            tokenizer_path=tokenizer_path
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # --- Load Dataset ---
    print("Loading dataset...")
    try:
        stories = load_dataset(dataset_path)
        if max_stories:
            stories = stories[:max_stories]
        print(f"Loaded {len(stories)} stories")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # For LLM Judge evaluation: randomly sample 1000 stories with seeding
    llm_judge_enabled = metrics_config.get("llm_judge", {}).get("enabled", False)
    llm_judge_stories_indices = None
    if llm_judge_enabled:
        if len(stories) > llm_judge_sample_size:
            random.seed(llm_judge_random_seed)
            llm_judge_stories_indices = set(sorted(random.sample(range(len(stories)), llm_judge_sample_size)))
            print(f"\nLLM Judge: Sampling {llm_judge_sample_size} stories (seed={llm_judge_random_seed}) for evaluation")
            print(f"  Selected story indices: first {sorted(list(llm_judge_stories_indices))[:10]}... (showing first 10)")
        else:
            llm_judge_stories_indices = set(range(len(stories)))
            print(f"\nLLM Judge: Using all {len(stories)} stories (less than {llm_judge_sample_size} available)")
    
    # --- Setup Optional Metrics ---
    metrics_config = config.get("metrics", {})
    
    text_quality_config = None
    text_quality_embedder = None
    if _HAS_TEXT_QUALITY and metrics_config.get("text_quality", {}).get("enabled", True):
        text_quality_config = TextQualityConfig(use_sentence_transformers=True)
        text_quality_embedder = _Embedder("all-MiniLM-L6-v2")
        # Move embedder to GPU for faster computation
        if hasattr(text_quality_embedder, 'to'):
            text_quality_embedder.to(device)
    
    noun_carryover_config = None
    noun_carryover_embedder = None
    if _HAS_NOUN_CARRYOVER and metrics_config.get("noun_carryover", {}).get("enabled", True):
        noun_carryover_config = SoftConfig()
        noun_carryover_embedder = SoftEmbedder("all-MiniLM-L6-v2")
        # Move embedder to GPU for faster computation
        if hasattr(noun_carryover_embedder, 'to'):
            noun_carryover_embedder.to(device)
    
    # --- Evaluate Stories ---
    print(f"\nEvaluating {len(stories)} stories...")
    all_results = []
    
    for idx, story in enumerate(stories):
        if idx % max(1, len(stories) // 10) == 0:
            print(f"  Progress: {idx}/{len(stories)}")
        
        # Split story
        prompt, ground_truth = split_story_at_midpoint(story)
        
        if not prompt or not ground_truth:
            print(f"  Skipping story {idx}: empty prompt or ground truth")
            continue
        
        # Generate
        try:
            generated = generate_from_prompt(
                model,
                tokenizer,
                prompt,
                model_type,
                generation_kwargs
            )
        except Exception as e:
            print(f"  Error generating for story {idx}: {e}")
            continue
        
        # Determine if this story should be evaluated with LLM judge
        eval_metrics_config = metrics_config.copy()
        if llm_judge_enabled and (llm_judge_stories_indices is None or idx not in llm_judge_stories_indices):
            # Disable LLM judge for this story
            eval_metrics_config["llm_judge"] = {"enabled": False}
        
        # Evaluate
        try:
            result = evaluate_story(
                prompt=prompt,
                generated=generated,
                ground_truth=ground_truth,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                device=device,
                metrics_config=eval_metrics_config,
                text_quality_config=text_quality_config,
                text_quality_embedder=text_quality_embedder,
                noun_carryover_config=noun_carryover_config,
                noun_carryover_embedder=noun_carryover_embedder,
            )
            result["model"] = model_type
            result["dataset"] = dataset_name
            result["story_id"] = idx
            all_results.append(result)
        except Exception as e:
            print(f"  Error evaluating story {idx}: {e}")
            continue
    
    if not all_results:
        print("ERROR: No stories were successfully evaluated")
        return
    
    print(f"\nSuccessfully evaluated {len(all_results)} stories\n")
    
    # --- Calculate Summary Statistics ---
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metric_keys = [k for k in all_results[0].keys() 
                   if k not in ["model", "dataset", "story_id", "prompt", "generated", "ground_truth"]]
    
    summary = {}
    for metric in metric_keys:
        values = [r[metric] for r in all_results if r[metric] is not None and not (isinstance(r[metric], float) and math.isnan(r[metric]))]
        if values:
            summary[metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
            print(f"\n{metric}:")
            if not math.isnan(summary[metric]['mean']):
                print(f"  Mean: {summary[metric]['mean']:.4f}")
                print(f"  Min:  {summary[metric]['min']:.4f}")
                print(f"  Max:  {summary[metric]['max']:.4f}")
                print(f"  Evaluated: {summary[metric]['count']}/{len(all_results)}")
            else:
                print(f"  Not computed (N/A for this model)")
        else:
            print(f"\n{metric}:")
            print(f"  Not computed for any stories")
    
    # --- Save JSON Results ---
    print("\n" + "="*80)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_output = {
        "metadata": {
            "model_path": str(model_path),
            "model_type": model_type,
            "dataset_path": str(dataset_path),
            "dataset_name": dataset_name,
            "total_stories_evaluated": len(all_results),
            "device": device,
        },
        "summary": summary,
        "results": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved results to {output_path}")
    
    # --- Log to W&B ---
    if HAS_WANDB:
        print("\nLogging to W&B...")
        wandb.init(
            project=wandb_project,
            name=f"model-eval_{Path(model_path).stem}_{dataset_name}",
            config={
                "model_path": str(model_path),
                "model_type": model_type,
                "tokenizer_type": tokenizer_type,
                "tokenizer_path": tokenizer_path or "default",
                "dataset_path": str(dataset_path),
                "dataset_name": dataset_name,
                "max_stories_evaluated": len(all_results),
            }
        )
        
        # Create results table
        table_data = []
        for result in all_results:
            row = [
                result["story_id"],
                model_type,
                dataset_name,
                result["prompt"],
                result["generated"],
                result["ground_truth"],
            ]
            
            # Add metric values
            for metric in metric_keys:
                val = result.get(metric)
                row.append(f"{val:.4f}" if val is not None and isinstance(val, float) else val)
            
            table_data.append(row)
        
        table = wandb.Table(
            columns=["story_id", "model", "dataset", "prompt", "generated", "ground_truth"] + metric_keys,
            data=table_data
        )
        
        wandb.log({
            f"{dataset_name}_results_table": table
        })
        
        # Log summary metrics
        summary_logs = {"total_stories_evaluated": len(all_results)}
        for metric, stats in summary.items():
            summary_logs[f"{metric}/mean"] = stats["mean"]
            summary_logs[f"{metric}/min"] = stats["min"]
            summary_logs[f"{metric}/max"] = stats["max"]
            summary_logs[f"{metric}/count"] = stats["count"]
        
        wandb.log(summary_logs)
        
        # Log JSON artifact
        with open("/tmp/evaluation_results.json", 'w') as f:
            json.dump(json_output, f, indent=2)
        
        # Sanitize artifact name (no spaces, parentheses, etc.)
        artifact_name = f"model_eval_{model_type}_{dataset_name}".replace(" ", "_").replace("(", "").replace(")", "")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation_results"
        )
        artifact.add_file("/tmp/evaluation_results.json", name="evaluation_results.json")
        wandb.log_artifact(artifact)
        
        # Log plots by metric category
        category_metrics = {
            "diversity": ["distinct_1", "distinct_2", "distinct_3"],
            "quality": ["grammar_score", "text_quality_score"],
            "overlap": ["bleu", "rouge_1", "rouge_2", "rouge_l"],
            "coherence": ["text_quality_coherence", "text_quality_cohesion"],
            "noun_carryover": ["noun_carryover_hard_coverage", "noun_carryover_hard_jaccard"],
        }
        
        for category, metrics in category_metrics.items():
            category_data = {m: summary[m]["mean"] for m in metrics if m in summary}
            if category_data:
                wandb.log(category_data)
        
        wandb.finish()
        print("Logged results to W&B")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
