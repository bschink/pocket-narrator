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
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import sys
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

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
    LLM_JUDGE_PROMPT_TEMPLATE,
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

# ============================================================================
# BATCH EVALUATION FUNCTION (GPU-optimized)
# ============================================================================

def evaluate_story_batch(
    batch_data: List[Tuple[int, str, str]],  # [(global_idx, prompt, completion), ...]
    device: str = "cpu",
    metrics_config: Optional[Dict] = None,
    text_quality_config: Optional['TextQualityConfig'] = None,
    text_quality_embedder: Optional['_Embedder'] = None,
    noun_carryover_config: Optional['SoftConfig'] = None,
    noun_carryover_embedder: Optional['SoftEmbedder'] = None,
    llm_judge_api_key: Optional[str] = None,
    llm_judge_stories_indices: Optional[set] = None,
) -> List[Dict]:
    """
    Batch evaluate multiple dataset stories with GPU optimization.
    
    Batches neural operations (embeddings) for efficiency.
    Sequential processing for expensive operations (LLM judge).
    
    Args:
        batch_data: List of (global_idx, prompt, completion) tuples
        device: Device for neural models
        metrics_config: Dictionary of enabled metrics
        
    Returns:
        List of result dictionaries
    """
    if not batch_data:
        return []
    
    results_list = []
    batch_size = len(batch_data)
    
    # Extract batch components
    prompts = [d[1] for d in batch_data]
    completions = [d[2] for d in batch_data]
    global_indices = [d[0] for d in batch_data]
    full_stories = [p + c for p, c in zip(prompts, completions)]
    
    # --- Pre-compute text quality embeddings if enabled ---
    tq_results_batch = [None] * batch_size
    if _HAS_TEXT_QUALITY and metrics_config.get("text_quality", {}).get("enabled", True):
        try:
            # Use indexed assignment to preserve batch alignment
            for batch_idx, completion in enumerate(completions):
                try:
                    tq_results = evaluate_text_quality(
                        completion,
                        cfg=text_quality_config,
                        embedder=text_quality_embedder
                    )
                    if isinstance(tq_results, dict):
                        tq_results_batch[batch_idx] = tq_results
                    else:
                        tq_results_batch[batch_idx] = None
                except Exception as e:
                    tq_results_batch[batch_idx] = None
        except Exception as e:
            print(f"    ⚠ Batch text quality failed: {e}")
            tq_results_batch = [None] * batch_size
    
    # --- Pre-compute noun carryover for all stories ---
    noun_carryover_batch = [None] * batch_size
    noun_carryover_enabled = metrics_config.get("noun_carryover", {}).get("enabled", True) if metrics_config else True
    noun_carryover_embedder = None
    if noun_carryover_enabled and _HAS_NOUN_CARRYOVER:
        # Initialize embedder once for efficiency
        noun_carryover_embedder = SoftEmbedder(SoftConfig(model_name="all-MiniLM-L6-v2"))
        try:
            for batch_idx, (prompt, completion) in enumerate(zip(prompts, completions)):
                if prompt:
                    soft_cfg = SoftConfig()
                    nc_results = noun_carryover_metrics(
                        prompt, 
                        completion, 
                        soft_cfg=soft_cfg,
                        embedder=noun_carryover_embedder
                    )
                    noun_carryover_batch[batch_idx] = nc_results
        except Exception as e:
            import traceback
            print(f"    ⚠ Batch noun carryover failed: {e}")
            traceback.print_exc()
            noun_carryover_batch = [None] * batch_size
    
    # --- Process each story in batch ---
    progress_iter = tqdm(
        enumerate(batch_data),
        total=len(batch_data),
        desc="  Stories",
        unit="story",
        ncols=100,
        leave=False
    ) if HAS_TQDM else enumerate(batch_data)
    
    for local_idx, (global_idx, prompt, completion) in progress_iter:
        try:
            full_story = full_stories[local_idx]
            results = {}
            
            # --- 1. Distinct-n ---
            try:
                results["distinct_1"] = distinct_n([full_story], n=1)
                results["distinct_2"] = distinct_n([full_story], n=2)
                results["distinct_3"] = distinct_n([full_story], n=3)
            except Exception as e:
                results["distinct_1"] = None
                results["distinct_2"] = None
                results["distinct_3"] = None
            
            # --- 2. Repetition rate ---
            try:
                results["repetition_rate"] = repetition_rate([full_story])
            except Exception as e:
                results["repetition_rate"] = None
            
            # --- 3. Grammar score (per-story) ---
            try:
                results["grammar_score"] = calculate_grammar_score([full_story], device=device)
            except Exception as e:
                results["grammar_score"] = None
            
            # --- 4. Word and sentence count ---
            try:
                results["word_count"] = count_words([full_story])
                results["sentence_count"] = count_sentences([full_story])
            except Exception as e:
                results["word_count"] = None
                results["sentence_count"] = None
            
            # --- 5. LLM Judge (handled separately, set to None here) ---
            results["llm_judge_grammar"] = None
            results["llm_judge_creativity"] = None
            results["llm_judge_consistency"] = None
            results["llm_judge_num_evaluated"] = None
            results["llm_judge_num_failed"] = None
            
            # --- 6. Text Quality (from pre-computed batch) ---
            if tq_results_batch[local_idx] and isinstance(tq_results_batch[local_idx], dict):
                try:
                    tq_results = tq_results_batch[local_idx]
                    results["text_quality_coherence"] = tq_results.get("coherence")
                    results["text_quality_cohesion"] = tq_results.get("cohesion_mean")
                    results["text_quality_score"] = tq_results.get("text_quality")
                except Exception as e:
                    results["text_quality_coherence"] = None
                    results["text_quality_cohesion"] = None
                    results["text_quality_score"] = None
            else:
                results["text_quality_coherence"] = None
                results["text_quality_cohesion"] = None
                results["text_quality_score"] = None
            
            # --- 7. Noun Carryover (from pre-computed batch) ---
            if noun_carryover_batch[local_idx] and isinstance(noun_carryover_batch[local_idx], dict):
                try:
                    nc_results = noun_carryover_batch[local_idx]
                    results["noun_carryover_hard_coverage"] = nc_results.get("hard_coverage")
                    results["noun_carryover_hard_jaccard"] = nc_results.get("hard_jaccard")
                    results["noun_carryover_hard_precision"] = nc_results.get("hard_precision")
                    results["noun_carryover_soft_coverage"] = nc_results.get("soft_coverage")
                    soft_cfg = noun_carryover_config if noun_carryover_config else SoftConfig()
                    results[f"noun_carryover_soft_coverage@{soft_cfg.threshold:.2f}"] = nc_results.get(
                        f"soft_coverage@{soft_cfg.threshold:.2f}"
                    )
                except Exception as e:
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
            
            results_list.append(results)
        except Exception as e:
            print(f"    ✗ Story {global_idx + 1}: {e}")
            continue
    
    return results_list


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
                       help="Maximum number of stories to evaluate in all phases (overrides config)")
    parser.add_argument("--llm_max_stories", type=int, default=None,
                       help="Maximum number of stories for LLM Judge Phase 3 sampling (overrides config)")
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
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for story evaluation (default: 16)")
    
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
    llm_max_stories = args.llm_max_stories or config.get("metrics", {}).get("llm_judge", {}).get("max_stories")
    device = args.device or config.get("device", {}).get("type", "cpu")
    output_json = args.output_json or config.get("output", {}).get("json_path")
    wandb_project = config.get("output", {}).get("wandb_project", "pocket-narrator")
    
    # LLM Judge sampling settings
    llm_judge_config = config.get("metrics", {}).get("llm_judge", {})
    llm_judge_sample_size = llm_judge_config.get("sample_size", 1000)
    llm_judge_random_seed = llm_judge_config.get("random_seed", 42)
    
    # Batch size configuration
    batch_size = args.batch_size or config.get("evaluation", {}).get("batch_size", 16)
    
    # W&B settings
    wandb_enabled = config.get("output", {}).get("wandb_enabled", True)
    
    # Metrics from config (CLI args take precedence)
    metrics_config = config.get("metrics", {})
    run_text_quality = args.run_text_quality is True or metrics_config.get("text_quality", {}).get("enabled", False)
    run_noun_carryover = args.run_noun_carryover is True or metrics_config.get("noun_carryover", {}).get("enabled", False)
    run_llm_judge = args.run_llm_judge is True or metrics_config.get("llm_judge", {}).get("enabled", False)
    llm_judge_api_key = args.llm_judge_api_key
    
    # --- Prompt for API Key if LLM Judge is Enabled ---
    if run_llm_judge and not llm_judge_api_key:
        print("\n" + "="*80)
        print("LLM Judge evaluation is enabled, but no API key provided.")
        print("Please enter your Google Gemini API key:")
        print("="*80)
        llm_judge_api_key = input("Google Gemini API Key: ").strip()
        if not llm_judge_api_key:
            print("ERROR: API key is required for LLM judge evaluation")
            return
    
    # Debug: Show metric settings
    print(f"Metrics Config - text_quality: {run_text_quality}, noun_carryover: {run_noun_carryover}, llm_judge: {run_llm_judge}")
    
    # Validate that dataset_path is provided
    if not dataset_path:
        parser.error("--dataset_path or --config with dataset.path is required")
    
    # Initialize W&B if available
    if HAS_WANDB and wandb_enabled:
        try:
            wandb.init(
                project=wandb_project,
                name=f"eval-{dataset_name}",
                config={
                    "dataset_path": dataset_path,
                    "dataset_name": dataset_name,
                    "max_stories": max_stories,
                    "llm_max_stories": llm_max_stories,
                    "device": device,
                    "run_text_quality": run_text_quality,
                    "run_noun_carryover": run_noun_carryover,
                    "run_llm_judge": run_llm_judge,
                }
            )
            print("✓ W&B initialized\n")
        except Exception as e:
            print(f"⚠ W&B initialization failed: {e}\n")
            wandb_enabled = False
            wandb_enabled = False
    
    print(f"Loading dataset from {dataset_path}...")
    stories = load_dataset(dataset_path)
    
    # Extract max_stories from CLI or config for truncating ALL evaluation
    max_stories = args.max_stories or config.get("dataset", {}).get("max_stories")
    if max_stories:
        stories = stories[:max_stories]
    
    print(f"Loaded {len(stories)} stories")
    print("Using midpoint splitting (50/50) matching training script prepare_batch()\n")
    
    # For LLM Judge evaluation: randomly sample stories with seeding
    llm_judge_stories_indices = None
    if run_llm_judge:
        # Use llm_max_stories if specified, otherwise use sample_size from config
        llm_sample_size = llm_max_stories if llm_max_stories else llm_judge_sample_size
        if len(stories) > llm_sample_size:
            random.seed(llm_judge_random_seed)
            llm_judge_stories_indices = set(sorted(random.sample(range(len(stories)), llm_sample_size)))
            print(f"LLM Judge: Sampling {llm_sample_size} stories (seed={llm_judge_random_seed}) for evaluation")
            print(f"  Selected story indices: first {sorted(list(llm_judge_stories_indices))[:10]}... (showing first 10)")
        else:
            llm_judge_stories_indices = set(range(len(stories)))
            print(f"LLM Judge: Using all {len(stories)} stories (less than {llm_sample_size} available)")
        print()
    
    # Evaluate all stories (with batching and phases)
    print("\n" + "="*80)
    print("NOTE: Dataset evaluation uses existing stories (no generation)")
    print("  Phase 1 (Generation): Not applicable - using ground truth stories")
    print("  Phase 2: Compute all metrics (diversity, grammar, text quality, etc.)")
    print("  Phase 3: LLM Judge evaluation (optional)")
    print("="*80 + "\n")
    
    print("Evaluating stories...")
    all_results = []
    print(f"Batch size: {batch_size}\n")
    
    # Initialize shared embedders for batch processing
    text_quality_embedder = None
    if _HAS_TEXT_QUALITY and run_text_quality:
        try:
            cfg = TextQualityConfig()
            if cfg.use_sentence_transformers:
                try:
                    text_quality_embedder = _Embedder(cfg.st_model)
                except Exception as e:
                    print(f"⚠ Text quality embedder initialization failed: {e}")
        except Exception as e:
            print(f"⚠ Text quality config error: {e}")
    
    noun_carryover_config = None
    if _HAS_NOUN_CARRYOVER and run_noun_carryover:
        try:
            noun_carryover_config = SoftConfig()
        except Exception as e:
            print(f"⚠ Noun carryover config error: {e}")
    
    text_quality_config = None
    if _HAS_TEXT_QUALITY and run_text_quality:
        try:
            text_quality_config = TextQualityConfig()
        except Exception as e:
            print(f"⚠ Text quality config error: {e}")
    
    metrics_config = {
        "text_quality": {"enabled": run_text_quality},
        "noun_carryover": {"enabled": run_noun_carryover},
        "llm_judge": {"enabled": run_llm_judge}
    }
    
    for batch_start in range(0, len(stories), batch_size):
        batch_end = min(batch_start + batch_size, len(stories))
        batch_stories = stories[batch_start:batch_end]
        
        # Progress header for this batch
        print(f"[BATCH {batch_start // batch_size + 1}] Processing stories {batch_start + 1}-{batch_end}/{len(stories)}")
        
        # Prepare batch data
        batch_data = []  # List of (idx, prompt, completion)
        
        for local_idx, story in enumerate(batch_stories):
            global_idx = batch_start + local_idx
            
            # Split story into prompt and completion
            prompt, completion = split_story_at_midpoint(story)
            
            if not completion:
                print(f"  ⚠ Story {global_idx + 1}: skipped (empty completion)")
                continue
            
            batch_data.append((global_idx, prompt, completion))
        
        # --- PHASE 2: NEURAL METRICS (GPU-batched) ---
        if batch_data:
            print(f"  └─ PHASE 2: Computing metrics for {len(batch_data)} stories (batched on GPU)...")
            
            try:
                # Process entire batch with GPU optimization
                batch_results = evaluate_story_batch(
                    batch_data,
                    device=device,
                    metrics_config=metrics_config,
                    text_quality_config=text_quality_config,
                    text_quality_embedder=text_quality_embedder,
                    noun_carryover_config=noun_carryover_config,
                    llm_judge_api_key=llm_judge_api_key,
                    llm_judge_stories_indices=llm_judge_stories_indices,
                )
                
                # Add metadata and collect results
                for result, (global_idx, prompt, completion) in zip(batch_results, batch_data):
                    result["dataset"] = dataset_name
                    result["story_id"] = global_idx
                    result["prompt"] = prompt
                    result["completion"] = completion
                    all_results.append(result)
                
                print(f"  │  ✓ Processed {len(batch_results)}/{len(batch_data)} stories")
            except Exception as e:
                print(f"  │  ✗ Batch evaluation failed: {e}")
                import traceback
                traceback.print_exc()
            
            print()  # blank line after batch
    
    
    if not all_results:
        print("ERROR: No stories were successfully evaluated")
        return
    
    print(f"✅ Successfully evaluated {len(all_results)} stories (Phases 1-3)\n")
    
    # --- PHASE 3: LLM Judge Evaluation (optional, expensive API calls) ---
    llm_judge_enabled = metrics_config.get("llm_judge", {}).get("enabled", False) if metrics_config else False
    if HAS_WANDB and llm_judge_enabled and llm_judge_stories_indices:
        print("="*80)
        print(f"PHASE 3: LLM Judge Evaluation ({len(llm_judge_stories_indices)} sampled stories)")
        print("="*80 + "\n")
        
        try:
            llm_judge_count = 0
            for story_idx in llm_judge_stories_indices:
                # Find result for this story
                result_idx = next((i for i, r in enumerate(all_results) if r["story_id"] == story_idx), None)
                if result_idx is None:
                    continue
                
                result = all_results[result_idx]
                story_num = story_idx + 1
                
                try:
                    print(f"\n  Story {story_num}:")
                    print(f"  {'─' * 78}")
                    
                    # Run LLM Judge
                    print(f"  🔄 Awaiting LLM Judge response...", end="", flush=True)
                    llm_results = run_llm_judge_evaluation(
                        [result["prompt"]], 
                        [result["completion"]], 
                        api_key=llm_judge_api_key
                    )
                    
                    if llm_results:
                        print(" ✓\n")
                        print(f"  📥 RESPONSE RECEIVED FROM LLM JUDGE:")
                        print(f"     Grammar Score: {llm_results.get('llm_judge_grammar')}")
                        print(f"     Creativity Score: {llm_results.get('llm_judge_creativity')}")
                        print(f"     Consistency Score: {llm_results.get('llm_judge_consistency')}")
                        print(f"     Evaluated: {llm_results.get('llm_judge_num_evaluated')}")
                        print(f"     Failed: {llm_results.get('llm_judge_num_failed')}")
                        
                        result["llm_judge_grammar"] = llm_results.get("llm_judge_grammar")
                        result["llm_judge_creativity"] = llm_results.get("llm_judge_creativity")
                        result["llm_judge_consistency"] = llm_results.get("llm_judge_consistency")
                        result["llm_judge_num_evaluated"] = llm_results.get("llm_judge_num_evaluated")
                        result["llm_judge_num_failed"] = llm_results.get("llm_judge_num_failed")
                        # Only count as successful if LLM judge successfully evaluated (not failed)
                        grammar = llm_results.get("llm_judge_grammar")
                        creativity = llm_results.get("llm_judge_creativity")
                        consistency = llm_results.get("llm_judge_consistency")
                        num_evaluated = llm_results.get("llm_judge_num_evaluated", 0)
                        num_failed = llm_results.get("llm_judge_num_failed", 1)
                        
                        # Check if evaluation was successful (num_evaluated > 0 and num_failed == 0)
                        if num_evaluated > 0 and num_failed == 0:
                            llm_judge_count += 1
                        else:
                            # Failed evaluation - print debug info
                            print(f"\n  ⚠️  EVALUATION FAILED")
                            print(f"  Grammar: {grammar}, Creativity: {creativity}, Consistency: {consistency}")
                            print(f"  Evaluated: {num_evaluated}, Failed: {num_failed}")
                            print(f"\n  🔍 DEBUG INFO:")
                            prompt_text = LLM_JUDGE_PROMPT_TEMPLATE.format(
                                story_beginning=result["prompt"],
                                story_completion=result["completion"]
                            )
                            print(f"  📤 FULL PROMPT SENT:")
                            print(f"  {prompt_text}")
                            print(f"\n  📥 RAW API RESPONSE:")
                            if "individual_scores" in llm_results and llm_results.get("individual_scores"):
                                for idx, score in enumerate(llm_results.get("individual_scores", [])):
                                    print(f"    Story {idx} raw response:")
                                    print(f"    {score.raw_response}")
                            else:
                                print(f"    No individual scores available")
                    else:
                        print(" ⚠\n")
                        print(f"  ⚠️  NO RESPONSE RECEIVED FROM LLM JUDGE")
                except Exception as e:
                    print(f" ✗\n")
                    print(f"  ❌ ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n{'═' * 80}")
            print(f"✅ LLM Judge evaluation complete ({llm_judge_count}/{len(llm_judge_stories_indices)} successful)\n")
        except Exception as e:
            print(f"\n⚠ LLM Judge phase encountered an error: {e}")
            print("Continuing with W&B logging (LLM judge results will be skipped)...\n")
            import traceback
            traceback.print_exc()
    elif run_llm_judge:
        print("\nℹ LLM Judge enabled but no stories sampled (check sample_size in config)\n")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metric_keys = [k for k in all_results[0].keys() 
                   if k not in ["dataset", "story_id", "prompt", "completion"]]
    
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
            print(f"  Mean: {summary[metric]['mean']:.4f}")
            print(f"  Min:  {summary[metric]['min']:.4f}")
            print(f"  Max:  {summary[metric]['max']:.4f}")
            print(f"  Evaluated: {summary[metric]['count']}/{len(all_results)}")
    
    # Log to W&B
    if HAS_WANDB and wandb_enabled:
        print("\n" + "="*80)
        print("LOGGING TO W&B")
        print("="*80 + "\n")
        
        try:
            # Create comprehensive results table with full data
            print(f"📊 Creating results table with {len(all_results)} stories...")
            columns = ["story_id", "model", "dataset", "prompt", "text"] + metric_keys
            table_data = []
            
            for result in all_results:
                row = [
                    result["story_id"],
                    "dataset",  # placeholder since we're evaluating a dataset
                    result["dataset"],
                    result["prompt"],
                    result["completion"],  # The story text being evaluated
                ]
                for metric in metric_keys:
                    value = result[metric]
                    if value is not None:
                        if isinstance(value, float):
                            row.append(f"{value:.4f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("N/A")
                table_data.append(row)
            
            # Log comprehensive results table
            wandb_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({"results_table": wandb_table})
            print(f"✓ Results table logged with {len(table_data)} rows")
            
            # Log summary metrics
            print("📈 Logging summary metrics...")
            summary_logs = {"total_stories_evaluated": len(all_results)}
            for metric, stats in summary.items():
                summary_logs[f"{metric}/mean"] = stats["mean"]
                summary_logs[f"{metric}/min"] = stats["min"]
                summary_logs[f"{metric}/max"] = stats["max"]
                summary_logs[f"{metric}/count"] = stats["count"]
            
            wandb.log(summary_logs)
            print(f"✓ Summary metrics logged ({len(summary_logs)} metrics)")
            
            # Log JSON artifact
            print("📁 Creating JSON artifact...")
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({
                    "dataset": dataset_name,
                    "total_stories": len(all_results),
                    "summary": summary,
                    "results": all_results
                }, f, indent=2, default=str)
                temp_json_path = f.name
            
            # Sanitize artifact name (no spaces, parentheses, etc.)
            artifact_name = f"dataset_eval_{dataset_name}".replace(" ", "_").replace("(", "").replace(")", "")
            results_artifact = wandb.Artifact(artifact_name, type="dataset")
            results_artifact.add_file(temp_json_path, name="evaluation_results.json")
            wandb.log_artifact(results_artifact)
            print(f"✓ JSON artifact logged: {artifact_name}")
            
            # Finish the run
            wandb.finish()
            print("✓ W&B run completed and uploaded")
            
        except Exception as e:
            print(f"✗ Error logging to W&B: {e}")
            import traceback
            traceback.print_exc()
            try:
                wandb.finish()
            except:
                pass
    else:
        if not HAS_WANDB:
            print("\nℹ Skipping W&B logging: wandb not installed")
        elif not wandb_enabled:
            print("\nℹ Skipping W&B logging: wandb_enabled=False in config")
    
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
    
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
