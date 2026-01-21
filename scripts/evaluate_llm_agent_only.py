"""
LLM Agent Evaluation Script

Evaluates only LLM-as-a-judge metrics on dataset stories and logs results to W&B.

LLM Judge Metrics:
  - llm_judge_grammar (1-3 scale)
  - llm_judge_creativity (1-3 scale)
  - llm_judge_consistency (1-3 scale)
  - llm_judge_age_groups (distribution of age groups A-F)

Usage:
    python scripts/evaluate_llm_agent_only.py \
        --dataset_path data/dataset.txt \
        --dataset_name "TinyStories Validation" \
        --api_key YOUR_API_KEY
        
    Or with config file:
    python scripts/evaluate_llm_agent_only.py --config configs/eval_config.yaml
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

from pocket_narrator.evaluate import run_llm_judge_evaluation


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

def evaluate_story_llm_only(
    prompt: str,
    completion: str,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate a single story with LLM judge evaluation only.
    
    Args:
        prompt: Story beginning/prompt (first half)
        completion: Generated story text (second half)
        api_key: API key for LLM judge
        
    Returns:
        Dictionary with LLM judge evaluation scores
    """
    results = {}
    
    # --- LLM Judge Evaluation ---
    try:
        llm_results = run_llm_judge_evaluation(
            story_beginnings=[prompt],
            story_completions=[completion],
            api_key=api_key,
            max_stories=1
        )
        results["llm_judge_grammar"] = llm_results.get("llm_judge_grammar")
        results["llm_judge_creativity"] = llm_results.get("llm_judge_creativity")
        results["llm_judge_consistency"] = llm_results.get("llm_judge_consistency")
        results["llm_judge_age_groups"] = llm_results.get("llm_judge_age_groups")
        results["llm_judge_num_evaluated"] = llm_results.get("llm_judge_num_evaluated")
        results["llm_judge_num_failed"] = llm_results.get("llm_judge_num_failed")
    except Exception as e:
        print(f"WARNING: LLM judge failed: {e}")
        results["llm_judge_grammar"] = None
        results["llm_judge_creativity"] = None
        results["llm_judge_consistency"] = None
        results["llm_judge_age_groups"] = None
        results["llm_judge_num_evaluated"] = None
        results["llm_judge_num_failed"] = None
    
    return results


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Agent-only dataset evaluation")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset file (overrides config)")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Name for W&B logging (overrides config)")
    parser.add_argument("--max_stories", type=int, default=None,
                       help="Maximum number of stories to evaluate (overrides config)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key for LLM judge (overrides config, uses GOOGLE_API_KEY env var if not provided)")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Save results to JSON file (overrides config)")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name (overrides config)")
    
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
    output_json = args.output_json or config.get("output", {}).get("json_path")
    wandb_project = args.wandb_project or config.get("output", {}).get("wandb_project", "pocket-narrator")
    
    # LLM Judge API key
    api_key = args.api_key or config.get("llm_judge", {}).get("api_key")
    if not api_key:
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
    
    # Validate that dataset_path is provided
    if not dataset_path:
        parser.error("--dataset_path or --config with dataset.path is required")
    
    # Warn if no API key
    if not api_key:
        print("WARNING: No API key provided. Set --api_key or GOOGLE_API_KEY environment variable.")
        print("Continuing without LLM judge evaluation...\n")
    
    # Initialize W&B if available
    if HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=f"llm-eval-{dataset_name}",
            config={
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "max_stories": max_stories,
                "evaluation_type": "llm_agent_only",
            }
        )
    
    print(f"Loading dataset from {dataset_path}...")
    stories = load_dataset(dataset_path)
    
    if max_stories:
        stories = stories[:max_stories]
    
    print(f"Loaded {len(stories)} stories")
    print("Using midpoint splitting (50/50) matching training script prepare_batch()\n")
    
    # Evaluate all stories
    print("Evaluating stories with LLM judge...")
    all_results = []
    
    for idx, story in enumerate(stories):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(stories)}")
        
        # Split story into prompt and completion
        prompt, completion = split_story_at_midpoint(story)
        
        if not completion:
            print(f"WARNING: Story {idx} has no completion after split, skipping")
            continue
        
        # Evaluate with LLM judge only
        eval_results = evaluate_story_llm_only(
            prompt=prompt,
            completion=completion,
            api_key=api_key,
        )
        
        # Package result with story info
        result_row = {
            "dataset": dataset_name,
            "story_id": idx,
            "prompt": prompt,
            "completion": completion,
            **eval_results
        }
        all_results.append(result_row)
    
    print(f"\nCompleted evaluation of {len(all_results)} stories")
    
    # Calculate summary statistics for numeric metrics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metric_keys = [k for k in all_results[0].keys() 
                   if k not in ["dataset", "story_id", "prompt", "completion"]]
    
    summary = {}
    for metric in metric_keys:
        if metric == "llm_judge_age_groups":
            # Handle age groups separately (it's a dictionary)
            age_groups = {}
            for result in all_results:
                if result[metric] is not None:
                    for age_group, count in result[metric].items():
                        age_groups[age_group] = age_groups.get(age_group, 0) + count
            
            summary[metric] = age_groups
            print(f"\n{metric}:")
            for age_group, count in sorted(age_groups.items()):
                print(f"  {age_group}: {count}")
        else:
            # Handle numeric metrics
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
        
        # Create unified results table that matches comprehensive eval structure
        # This allows all evaluation types (dataset_eval, model_eval, llm_eval) to share one table
        unified_columns = [
            "story_id", "dataset", "prompt", "completion",
            # Diversity metrics
            "distinct_1", "distinct_2", "distinct_3",
            # Repetition
            "repetition_rate",
            # Grammar
            "grammar_score",
            # Word/sentence counts
            "word_count", "sentence_count",
            # Text quality
            "text_quality_coherence", "text_quality_cohesion", "text_quality_score",
            # Noun carryover
            "noun_carryover_hard_coverage", "noun_carryover_hard_jaccard", 
            "noun_carryover_hard_precision", "noun_carryover_soft_coverage",
            # LLM Judge metrics
            "llm_judge_grammar", "llm_judge_creativity", "llm_judge_consistency",
            "llm_judge_age_groups", "llm_judge_num_evaluated", "llm_judge_num_failed"
        ]
        
        table_data = []
        for result in all_results:
            row = [
                result["story_id"],
                result["dataset"],
                result["prompt"][:100],  # Truncate for display
                result["completion"][:100],  # Truncate for display
            ]
            
            # Add all other metrics (N/A if not computed by this script)
            for col in unified_columns[4:]:  # Skip story_id, dataset, prompt, completion
                if col in result:
                    value = result[col]
                    if col == "llm_judge_age_groups":
                        row.append(str(value) if value is not None else "N/A")
                    else:
                        row.append(value if value is not None else "N/A")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        # Log to unified evaluation table (same as evaluate_model.py)
        wandb_table = wandb.Table(columns=unified_columns, data=table_data)
        wandb.log({
            f"{dataset_name}_results_table": wandb_table
        })
        
        # Also log as artifacts for easy download
        import tempfile
        results_artifact = wandb.Artifact("llm_evaluation_results", type="dataset")
        
        # Save detailed results to temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "dataset": dataset_name,
                "total_stories": len(all_results),
                "summary": summary,
                "results": all_results
            }, f, indent=2, default=str)
            temp_json_path = f.name
        
        # Add to artifact
        results_artifact.add_file(temp_json_path, name="llm_evaluation_results.json")
        wandb.log_artifact(results_artifact)
        
        # Log summary metrics
        summary_logs = {"total_stories_evaluated": len(all_results)}
        for metric, stats in summary.items():
            if metric == "llm_judge_age_groups":
                for age_group, count in stats.items():
                    summary_logs[f"age_group_{age_group}"] = count
            else:
                summary_logs[f"{metric}/mean"] = stats["mean"]
                summary_logs[f"{metric}/min"] = stats["min"]
                summary_logs[f"{metric}/max"] = stats["max"]
                summary_logs[f"{metric}/count"] = stats["count"]
        
        wandb.log(summary_logs)
        
        # Create visualization plots for LLM metrics
        llm_plot_data = []
        for result in all_results:
            for metric in ["llm_judge_grammar", "llm_judge_creativity", "llm_judge_consistency"]:
                if metric in result and result[metric] is not None:
                    llm_plot_data.append({
                        "metric": metric,
                        "value": result[metric],
                        "story_id": result["story_id"]
                    })
        
        if llm_plot_data:
            import pandas as pd
            wandb.log({"llm_judge_metrics_plot": wandb.Table(dataframe=pd.DataFrame(llm_plot_data))})
        
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
