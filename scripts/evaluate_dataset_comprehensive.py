"""
Comprehensive Dataset Evaluation Script

Evaluates all available metrics on dataset stories and logs results to W&B.

Metrics evaluated:
  - distinct_n (1, 2, 3)
  - repetition_rate
  - grammar_score
  - llm_judge (grammar, creativity, consistency)
  - text_quality (coherence, cohesion)
  - noun_carryover (hard_coverage, hard_jaccard, hard_precision, soft_coverage)
  - word_count
  - sentence_count

Usage:
    python scripts/evaluate_dataset_comprehensive.py \
        --dataset_path data/dataset.txt \
        --dataset_name "TinyStories Validation" \
        --split_at_first_sentence
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

from pocket_narrator.evaluate import (
    distinct_n,
    repetition_rate,
    calculate_grammar_score,
    count_words,
    count_sentences,
    run_llm_judge_evaluation,
    _HAS_TEXT_QUALITY,
    _HAS_NOUN_CARRYOVER,
)

if _HAS_TEXT_QUALITY:
    from pocket_narrator.text_quality import TextQualityConfig, evaluate_text_quality, _Embedder

if _HAS_NOUN_CARRYOVER:
    from pocket_narrator.noun_carryover import noun_carryover_metrics, SoftConfig


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
        (prompt, completion) tuple
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


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_story(
    story: str,
    prompt: str,
    device: str = "cpu",
    run_llm_judge: bool = False,
    run_text_quality: bool = True,
    run_noun_carryover: bool = True,
    llm_judge_api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate a single story with all available metrics.
    
    Args:
        story: Generated story text
        prompt: Story beginning/prompt
        device: Device for grammar evaluation ("cpu", "cuda", "mps")
        run_llm_judge: Whether to run LLM judge evaluation
        run_text_quality: Whether to run text quality evaluation
        run_noun_carryover: Whether to run noun carryover evaluation
        llm_judge_api_key: API key for LLM judge
        
    Returns:
        Dictionary with all evaluation scores
    """
    results = {}
    
    # --- 1. Distinct-n ---
    for n in (1, 2, 3):
        results[f"distinct_{n}"] = distinct_n([story], n=n)
    
    # --- 2. Repetition rate ---
    results["repetition_rate"] = repetition_rate([story])
    
    # --- 3. Grammar score ---
    try:
        results["grammar_score"] = calculate_grammar_score([story], device=device)
    except Exception as e:
        print(f"WARNING: Grammar evaluation failed: {e}")
        results["grammar_score"] = None
    
    # --- 4. Word and sentence count ---
    results["word_count"] = count_words([story])
    results["sentence_count"] = count_sentences([story])
    
    # --- 5. LLM Judge (optional) ---
    if run_llm_judge:
        try:
            llm_results = run_llm_judge_evaluation(
                story_beginnings=[prompt],
                story_completions=[story],
                api_key=llm_judge_api_key,
                max_stories=1
            )
            results["llm_judge_grammar"] = llm_results.get("llm_judge_grammar")
            results["llm_judge_creativity"] = llm_results.get("llm_judge_creativity")
            results["llm_judge_consistency"] = llm_results.get("llm_judge_consistency")
            results["llm_judge_num_evaluated"] = llm_results.get("llm_judge_num_evaluated")
            results["llm_judge_num_failed"] = llm_results.get("llm_judge_num_failed")
        except Exception as e:
            print(f"WARNING: LLM judge failed: {e}")
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
    
    # --- 6. Text Quality (optional) ---
    if run_text_quality and _HAS_TEXT_QUALITY:
        try:
            cfg = TextQualityConfig()
            embedder = None
            if cfg.use_sentence_transformers:
                try:
                    embedder = _Embedder(cfg.st_model)
                except Exception:
                    pass
            
            metrics = evaluate_text_quality(story, cfg=cfg, embedder=embedder)
            results["text_quality_coherence"] = metrics.get("coherence")
            results["text_quality_cohesion"] = metrics.get("cohesion_mean")
            results["text_quality_score"] = metrics.get("text_quality")
        except Exception as e:
            print(f"WARNING: Text quality evaluation failed: {e}")
            results["text_quality_coherence"] = None
            results["text_quality_cohesion"] = None
            results["text_quality_score"] = None
    else:
        results["text_quality_coherence"] = None
        results["text_quality_cohesion"] = None
        results["text_quality_score"] = None
    
    # --- 7. Noun Carryover (optional) ---
    if run_noun_carryover and _HAS_NOUN_CARRYOVER and prompt:
        try:
            soft_cfg = SoftConfig()
            metrics = noun_carryover_metrics(
                prompt,
                story,
                soft_cfg=soft_cfg
            )
            results["noun_hard_coverage"] = metrics.get("hard_coverage")
            results["noun_hard_jaccard"] = metrics.get("hard_jaccard")
            results["noun_hard_precision"] = metrics.get("hard_precision")
            results["noun_soft_coverage"] = metrics.get("soft_coverage")
            results[f"noun_soft_coverage@{soft_cfg.threshold:.2f}"] = metrics.get(
                f"soft_coverage@{soft_cfg.threshold:.2f}"
            )
        except Exception as e:
            print(f"WARNING: Noun carryover evaluation failed: {e}")
            results["noun_hard_coverage"] = None
            results["noun_hard_jaccard"] = None
            results["noun_hard_precision"] = None
            results["noun_soft_coverage"] = None
            results[f"noun_soft_coverage@{0.70:.2f}"] = None
    else:
        results["noun_hard_coverage"] = None
        results["noun_hard_jaccard"] = None
        results["noun_hard_precision"] = None
        results["noun_soft_coverage"] = None
        results[f"noun_soft_coverage@{0.70:.2f}"] = None
    
    return results


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset evaluation")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset file (overrides config)")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for W&B logging (overrides config)")
    parser.add_argument("--max_stories", type=int, default=None,
                       help="Maximum number of stories to evaluate (overrides config)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                       help="Device for neural models (overrides config)")
    parser.add_argument("--run_llm_judge", action="store_true", default=None,
                       help="Run LLM judge evaluation (overrides config)")
    parser.add_argument("--llm_judge_api_key", type=str, default=None,
                       help="API key for LLM judge (overrides config)")
    parser.add_argument("--run_text_quality", action="store_true", default=None,
                       help="Run text quality evaluation (overrides config)")
    parser.add_argument("--run_noun_carryover", action="store_true", default=None,
                       help="Run noun carryover evaluation (overrides config)")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Save results to JSON file (overrides config)")
    
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
    
    # Extract values from config with CLI overrides
    dataset_path = args.dataset_path or config.get("dataset", {}).get("path")
    dataset_name = args.dataset_name or config.get("dataset", {}).get("name", "Dataset")
    max_stories = args.max_stories or config.get("dataset", {}).get("max_stories")
    device = args.device or config.get("device", {}).get("type", "cpu")
    output_json = args.output_json or config.get("output", {}).get("json_path")
    wandb_project = config.get("output", {}).get("wandb_project", "pocket-narrator")
    
    # Metrics from config (CLI args take precedence)
    metrics_config = config.get("metrics", {})
    run_text_quality = args.run_text_quality is True or metrics_config.get("text_quality", {}).get("enabled", False)
    run_noun_carryover = args.run_noun_carryover is True or metrics_config.get("noun_carryover", {}).get("enabled", False)
    run_llm_judge = args.run_llm_judge is True or metrics_config.get("llm_judge", {}).get("enabled", False)
    llm_judge_api_key = args.llm_judge_api_key or metrics_config.get("llm_judge", {}).get("api_key")
    
    # Validate that dataset_path is provided
    if not dataset_path:
        parser.error("--dataset_path or --config with dataset.path is required")
    
    # Initialize W&B if available
    if HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"eval-{dataset_name}",
            config={
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "max_stories": max_stories,
                "device": device,
                "run_text_quality": run_text_quality,
                "run_noun_carryover": run_noun_carryover,
                "run_llm_judge": run_llm_judge,
            }
        )
    
    print(f"Loading dataset from {dataset_path}...")
    stories = load_dataset(dataset_path)
    
    if max_stories:
        stories = stories[:max_stories]
    
    print(f"Loaded {len(stories)} stories")
    print("Using midpoint splitting (50/50) matching training script prepare_batch()\n")
    
    # Evaluate all stories
    print("Evaluating stories...")
    all_results = []
    
    for idx, story in enumerate(stories):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(stories)}")
        
        # Split story into prompt and completion
        prompt, completion = split_story_at_midpoint(story)
        
        if not completion:
            print(f"WARNING: Story {idx} has no completion after split, skipping")
            continue
        
        # Evaluate
        eval_results = evaluate_story(
            story=completion,
            prompt=prompt,
            device=device,
            run_llm_judge=run_llm_judge,
            run_text_quality=run_text_quality,
            run_noun_carryover=run_noun_carryover,
            llm_judge_api_key=llm_judge_api_key,
        )
        
        # Package result with story info
        result_row = {
            "dataset": dataset_name,
            "story_id": idx,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "completion": completion[:100] + "..." if len(completion) > 100 else completion,
            **eval_results
        }
        all_results.append(result_row)
    
    print(f"\nCompleted evaluation of {len(all_results)} stories")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metric_keys = [k for k in all_results[0].keys() 
                   if k not in ["dataset", "story_id", "prompt", "completion"]]
    
    summary = {}
    for metric in metric_keys:
        values = [r[metric] for r in all_results if r[metric] is not None]
        if values:
            summary[metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
            print(f"\n{metric}:")
            print(f"  Mean: {summary[metric]['mean']:.4f}")
            print(f"  Min:  {summary[metric]['min']:.4f}")
            print(f"  Max:  {summary[metric]['max']:.4f}")
            print(f"  Evaluated: {summary[metric]['count']}/{len(all_results)}")
    
    # Log to W&B
    if HAS_WANDB:
        print("\nLogging to W&B...")
        
        # Create comprehensive results table with full data
        columns = ["story_id", "prompt", "completion"] + metric_keys
        table_data = []
        
        for result in all_results:
            row = [
                result["story_id"],
                result["prompt"],
                result["completion"],
            ]
            for metric in metric_keys:
                value = result[metric]
                row.append(value if value is not None else "N/A")
            table_data.append(row)
        
        # Log comprehensive results table
        wandb_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({f"{dataset_name}_results_table": wandb_table})
        
        # Also log as artifacts for easy download
        import json
        results_artifact = wandb.Artifact("evaluation_results", type="dataset")
        
        # Save detailed results to temp JSON
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "dataset": dataset_name,
                "total_stories": len(all_results),
                "summary": summary,
                "results": all_results
            }, f, indent=2, default=str)
            temp_json_path = f.name
        
        # Add to artifact
        results_artifact.add_file(temp_json_path, name="evaluation_results.json")
        wandb.log_artifact(results_artifact)
        
        # Log summary metrics
        summary_logs = {"total_stories_evaluated": len(all_results)}
        for metric, stats in summary.items():
            summary_logs[f"{metric}/mean"] = stats["mean"]
            summary_logs[f"{metric}/min"] = stats["min"]
            summary_logs[f"{metric}/max"] = stats["max"]
            summary_logs[f"{metric}/count"] = stats["count"]
        
        wandb.log(summary_logs)
        
        # Create visualization plots for each metric category
        metric_categories = {
            "diversity": ["distinct_1", "distinct_2", "distinct_3"],
            "repetition": ["repetition_rate"],
            "grammar": ["grammar_score"],
            "text_quality": ["text_quality_coherence", "text_quality_cohesion", "text_quality_score"],
            "noun_carryover": ["noun_hard_coverage", "noun_hard_jaccard", "noun_hard_precision", 
                              "noun_soft_coverage", "noun_soft_coverage@0.70"],
            "llm_judge": ["llm_judge_grammar", "llm_judge_creativity", "llm_judge_consistency"],
            "statistics": ["word_count", "sentence_count"],
        }
        
        for category, metrics in metric_categories.items():
            category_data = []
            for result in all_results:
                for metric in metrics:
                    if metric in result and result[metric] is not None:
                        category_data.append({
                            "metric": metric,
                            "value": result[metric],
                            "story_id": result["story_id"]
                        })
            
            if category_data:
                wandb.log({f"{category}_plot": wandb.Table(dataframe=__import__('pandas').DataFrame(category_data))})
        
        wandb.finish()
    
    # Save to JSON if requested
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": dataset_name,
                "total_stories": len(all_results),
                "summary": summary,
                "results": all_results
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
