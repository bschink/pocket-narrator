"""
Compare multiple GPT-2 models (small, medium, large, xlarge) trained on TinyStories.

The script:
- Loads each model (HuggingFace-style) and its tokenizer
- Reads vocab_size and context length (n_ctx / model_max_length)
- Evaluates:
    * validation loss + perplexity on TinyStories validation text
    * lexical diversity (Distinct-1/2/3) on generated samples
    * repetition rate (3-gram) on generated samples

Usage example:

    ./.venv/bin/python gpt2/compare_gpt2_models.py \
        --models \
          small:results/gpt2_small \
          medium:results/gpt2_medium \
          large:results/gpt2_large\
          xlarge:results/gpt2_x_large \
        --validation data/TinyStories-valid.txt \
        --max_new_tokens 80 \
        --temperature 0.7 \
        --device mps

Notes:
- Each model_root must be a directory that `GPT2LMHeadModel.from_pretrained(model_root)` can load.
- Tokenizer is loaded from the same directory (or from `model_root/tokenizer` if present).
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import os
from pathlib import Path


# ----------------------------------------------------------------------
# Simple tokenization & metrics
# ----------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()


def distinct_n(texts: List[str], n: int = 1) -> float:
    """
    Distinct-n: |unique n-grams| / |all n-grams|.
    """
    from collections import Counter

    all_ngrams = []
    for t in texts:
        tokens = simple_tokenize(t)
        if len(tokens) < n:
            continue
        all_ngrams.extend(
            tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
        )
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    return len(counts) / float(len(all_ngrams))


def repetition_rate(texts: List[str], n: int = 3) -> float:
    """
    Proportion of n-grams that occur more than once (average across texts)
    """
    from collections import Counter

    rates: List[float] = []
    for t in texts:
        tokens = simple_tokenize(t)
        if len(tokens) < n:
            rates.append(0.0)
            continue
        ngrams = [
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        ]
        if not ngrams:
            rates.append(0.0)
            continue
        counts = Counter(ngrams)
        repeated = sum(c for c in counts.values() if c > 1)
        total = sum(counts.values())
        rates.append(repeated / total if total > 0 else 0.0)
    return sum(rates) / len(rates) if rates else 0.0


# ----------------------------------------------------------------------
# Device handling
# ----------------------------------------------------------------------

def choose_device(name: Optional[str]) -> torch.device:
    if name:
        name = name.lower()
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Loading model & tokenizer (robust to different layouts)
# ----------------------------------------------------------------------

def _find_dir_with_files(root: Path, required_any: list[str], max_depth: int = 3) -> Path | None:
    """
   
    """
    root = root.resolve()

    # 1) Root 
    if any((root / f).exists() for f in required_any):
        return root

    # 2) Recursive
    for dirpath, dirnames, filenames in os.walk(root):
        rel_parts = Path(dirpath).relative_to(root).parts
        if len(rel_parts) > max_depth:
            continue
        if any(f in filenames for f in required_any):
            return Path(dirpath)

    return None


def load_model_and_tokenizer(
    model_root: str,
    device: torch.device,
) -> Tuple[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    """
    Lädt Modell & Tokenizer möglichst robust.

    Unterstützt u.a. zwei Fälle:

    1) Dein altes Setup:
        gpt2/
          Gpt2Model/       <- model_dir
          GPT2Tokenize/    <- tok_dir

       Aufruf dann: --models small:gpt2

    2) Neue Runs:
        gpt2/results/gpt2_small/
          model/           <- or direct config.json, pytorch_model.bin
          tokenizer/

       Aufruf: --models small:gpt2/results/gpt2_small
    """
    root = Path(model_root).resolve()

    # s Layout gpt2/Gpt2Model + gpt2/GPT2Tokenize
    gpt2model_dir = root / "Gpt2Model"
    gpt2token_dir = root / "GPT2Tokenize"
    if gpt2model_dir.exists() and gpt2token_dir.exists():
        model_dir = gpt2model_dir
        tok_dir = gpt2token_dir
    else:
        # Allgemeiner Fall: irgendwo unterhalb von root ein HF-Model suchen
        model_dir = _find_dir_with_files(
            root,
            required_any=["config.json", "pytorch_model.bin", "model.safetensors"],
            max_depth=3,
        )
        if model_dir is None:
            raise ValueError(f"Could not find a valid model directory under {root}")

        tok_dir = _find_dir_with_files(
            root,
            required_any=["tokenizer.json", "vocab.json", "merges.txt"],
            max_depth=3,
        )
        if tok_dir is None:
            # Fallback: Tokenizer in own like model
            tok_dir = model_dir
            if not any((tok_dir / f).exists() for f in ["tokenizer.json", "vocab.json", "merges.txt"]):
                raise ValueError(f"Could not find a valid tokenizer directory under {root}")

    print(f"  Loading model from {model_dir}")
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    print(f"  Loading tokenizer from {tok_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)

    model.to(device)
    model.eval()
    return model, tokenizer



# ----------------------------------------------------------------------
# Validation perplexity
# ----------------------------------------------------------------------

def compute_validation_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    val_lines: List[str],
    device: torch.device,
    max_samples: int = 256,
    max_seq_len: int = 256,
) -> Dict[str, Optional[float]]:
    if not val_lines:
        return {"val_loss": None, "val_perplexity": None}

    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for line in val_lines[:max_samples]:
            enc = tokenizer(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, labels=enc["input_ids"])
            losses.append(float(outputs.loss.item()))

    if not losses:
        return {"val_loss": None, "val_perplexity": None}

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"val_loss": mean_loss, "val_perplexity": ppl}


# ----------------------------------------------------------------------
# Generation & metrics
# ----------------------------------------------------------------------

def generate_texts(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
) -> List[str]:
    torch.manual_seed(seed)
    texts: List[str] = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-8),
                top_k=top_k if top_k > 0 else 0,
                top_p=top_p,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        texts.append(text)

    return texts


def compute_generation_stats(generated_texts: List[str]) -> Dict[str, float]:
    return {
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
        "distinct_3": distinct_n(generated_texts, 3),
        "repetition_rate": repetition_rate(generated_texts, 3),
    }


# ----------------------------------------------------------------------
# CLI & main
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple GPT-2 models (small/medium/large/xlarge) on TinyStories."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "List of name:path pairs, e.g. "
            "small:results/gpt2_small_2k_ctx256 medium:results/gpt2_medium_8k_ctx256"
        ),
    )
    parser.add_argument(
        "--validation",
        type=str,
        required=True,
        help="Path to TinyStories validation file (one story per line).",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=256,
        help="Number of validation lines to use for perplexity.",
    )
    parser.add_argument(
        "--num_gen_prompts",
        type=int,
        default=16,
        help="Number of prompts (sampled from validation) for generation/diversity metrics.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Max new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter (0 = disabled).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    # Parse models: name:path
    models_spec: Dict[str, str] = {}
    for spec in args.models:
        if ":" not in spec:
            raise ValueError(f"Model spec must be name:path, got: {spec}")
        name, path = spec.split(":", 1)
        models_spec[name] = path

    # Load validation data
    val_path = Path(args.validation)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    with val_path.open("r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(val_lines)} validation lines from {val_path}")

    # Sample prompts for generation
    num_gen = min(args.num_gen_prompts, len(val_lines))
    gen_prompts = random.sample(val_lines, k=num_gen)
    print(f"Using {num_gen} prompts sampled from validation for generation metrics.")

    summaries: Dict[str, Dict[str, float]] = {}

    # Evaluate each model
    for name, model_root in models_spec.items():
        print("\n" + "-" * 80)
        print(f"Evaluating model: {name} ({model_root})")
        start_time = time.time()

        model, tokenizer = load_model_and_tokenizer(model_root, device)

        # Read vocab & context length from model/tokenizer
        vocab_size = tokenizer.vocab_size
        max_ctx = getattr(model.config, "n_ctx", None) or getattr(model.config, "n_positions", None)
        tok_ctx = getattr(tokenizer, "model_max_length", None)

        # Validation perplexity
        val_stats = compute_validation_perplexity(
            model=model,
            tokenizer=tokenizer,
            val_lines=val_lines,
            device=device,
            max_samples=args.num_val_samples,
            max_seq_len=min(max_ctx or 256, 512),
        )

        # Generation & diversity metrics
        generated = generate_texts(
            model=model,
            tokenizer=tokenizer,
            prompts=gen_prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        gen_stats = compute_generation_stats(generated)

        elapsed = time.time() - start_time
        num_params = sum(p.numel() for p in model.parameters())

        summary = {
            "vocab_size": float(vocab_size),
            "model_ctx": float(max_ctx) if max_ctx is not None else float("nan"),
            "tokenizer_ctx": float(tok_ctx) if tok_ctx is not None else float("nan"),
            "num_params": float(num_params),
            "val_loss": val_stats["val_loss"] if val_stats["val_loss"] is not None else float("nan"),
            "val_perplexity": val_stats["val_perplexity"] if val_stats["val_perplexity"] is not None else float("nan"),
            "distinct_1": gen_stats["distinct_1"],
            "distinct_2": gen_stats["distinct_2"],
            "distinct_3": gen_stats["distinct_3"],
            "repetition_rate": gen_stats["repetition_rate"],
            "eval_time_seconds": elapsed,
        }
        summaries[name] = summary

        # Print per-model summary
        print(f"Model: {name}")
        print(f"  vocab_size          : {vocab_size}")
        print(f"  model_ctx (n_ctx)   : {max_ctx}")
        print(f"  tokenizer_ctx       : {tok_ctx}")
        print(f"  num_params          : {num_params:,}")
        print(f"  val_loss            : {summary['val_loss']:.4f}")
        print(f"  val_perplexity      : {summary['val_perplexity']:.4f}")
        print(f"  distinct_1          : {summary['distinct_1']:.4f}")
        print(f"  distinct_2          : {summary['distinct_2']:.4f}")
        print(f"  distinct_3          : {summary['distinct_3']:.4f}")
        print(f"  repetition_rate     : {summary['repetition_rate']:.4f}")
        print(f"  eval_time_seconds   : {summary['eval_time_seconds']:.2f}")

    # Print global comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    header = (
        f"{'model':<12} | {'vocab':>7} | {'ctx':>5} | {'params(M)':>9} | "
        f"{'ppl':>8} | {'dist1':>6} | {'dist2':>6} | {'rept':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        params_m = s["num_params"] / 1e6
        ctx = int(s["model_ctx"]) if not math.isnan(s["model_ctx"]) else 0
        print(
            f"{name:<12} | "
            f"{int(s['vocab_size']):>7d} | "
            f"{ctx:>5d} | "
            f"{params_m:>9.2f} | "
            f"{s['val_perplexity']:>8.2f} | "
            f"{s['distinct_1']:>6.3f} | "
            f"{s['distinct_2']:>6.3f} | "
            f"{s['repetition_rate']:>6.3f}"
        )

    # Optionally dump JSON for later analysis
    out_path = Path("gpt2_model_comparison_summary.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved detailed summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
