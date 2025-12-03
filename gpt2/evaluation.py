#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import nltk
nltk.download('punkt')
from config_utils import load_yaml
from typing import Optional


import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# Optional metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False



# Helpers


def load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_texts(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    prompts: List[str],
    max_gen_length: int,
    device: torch.device,
    batch_size: int = 4,
) -> List[str]:
    model.eval()
    outputs: List[str] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for g in gen_ids:
                text = tokenizer.decode(g, skip_special_tokens=True)
                outputs.append(text)

    return outputs


def _tokenize_for_metrics(text: str) -> List[str]:
    # simpler Tokenisierung für Metriken
    return text.strip().split()


def repetition_rate(text: str, n: int = 3) -> float:
    """
    Anteil der n-Gramme, die mehr als einmal vorkommen.
    0.0 = keine Wiederholung, 1.0 = alles nur Wiederholung.
    """
    tokens = _tokenize_for_metrics(text)
    if len(tokens) < n + 1:
        return 0.0

    ngrams = [
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    ]
    if not ngrams:
        return 0.0

    from collections import Counter

    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    total = sum(counts.values())
    return repeated / total if total > 0 else 0.0


def length_stats(gens: List[str], refs: Optional[List[str]] = None) -> Dict[str, Any]:
    gen_lens = [len(_tokenize_for_metrics(g)) for g in gens]
    stats: Dict[str, Any] = {
        "gen_len_mean": sum(gen_lens) / len(gen_lens) if gen_lens else 0.0,
        "gen_len_min": min(gen_lens) if gen_lens else 0,
        "gen_len_max": max(gen_lens) if gen_lens else 0,
    }
    if refs is not None:
        ref_lens = [len(_tokenize_for_metrics(r)) for r in refs]
        # Simple Length-Penalty: mittlere relative Abweichung
        penalties = []
        for gl, rl in zip(gen_lens, ref_lens):
            if rl > 0:
                penalties.append(abs(gl - rl) / rl)
        stats["length_penalty_mean"] = (
            sum(penalties) / len(penalties) if penalties else 0.0
        )
        stats["ref_len_mean"] = (
            sum(ref_lens) / len(ref_lens) if ref_lens else 0.0
        )
    return stats



# Metric computation


def compute_bleu(
    refs: List[str],
    hyps: List[str],
) -> Optional[float]:
    if not _HAS_NLTK:
        print("NLTK not installed, skipping BLEU.")
        return None

    smooth_fn = SmoothingFunction().method1
    scores = []
    for r, h in zip(refs, hyps):
        ref_tokens = _tokenize_for_metrics(r)
        hyp_tokens = _tokenize_for_metrics(h)
        if not hyp_tokens:
            continue
        scores.append(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=smooth_fn,
            )
        )
    return sum(scores) / len(scores) if scores else None


def compute_rouge_l(
    refs: List[str],
    hyps: List[str],
) -> Optional[Dict[str, float]]:
    if not _HAS_ROUGE:
        print("rouge_score not installed, skipping ROUGE-L.")
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f_scores = []
    p_scores = []
    r_scores = []

    for r, h in zip(refs, hyps):
        res = scorer.score(r, h)["rougeL"]
        f_scores.append(res.fmeasure)
        p_scores.append(res.precision)
        r_scores.append(res.recall)

    if not f_scores:
        return None

    return {
        "rougeL_f": sum(f_scores) / len(f_scores),
        "rougeL_p": sum(p_scores) / len(p_scores),
        "rougeL_r": sum(r_scores) / len(r_scores),
    }


def compute_bertscore(
    refs: List[str],
    hyps: List[str],
    model_type: str = "microsoft/deberta-base-mnli",
) -> Optional[Dict[str, float]]:
    if not _HAS_BERTSCORE:
        print("bert_score not installed, skipping BERTScore.")
        return None

    # BertScore erwartet Lists-of-strings
    P, R, F1 = bert_score(
        cands=hyps,
        refs=refs,
        model_type=model_type,
        lang="en",
        verbose=True,
    )
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_repetition_stats(hyps: List[str]) -> Dict[str, float]:
    rates = [repetition_rate(h) for h in hyps]
    return {
        "repetition_mean": sum(rates) / len(rates) if rates else 0.0,
        "repetition_min": min(rates) if rates else 0.0,
        "repetition_max": max(rates) if rates else 0.0,
    }


# Main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyStories LM outputs.")

    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Optional YAML file with evaluation settings.",
    )

    # Die restlichen Argumente sind jetzt NICHT mehr required,
    # weil sie durch eval-config überschrieben werden können.
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory with trained model (trainer.save_model).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Directory with saved tokenizer.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Text file: one prompt per line.",
    )
    parser.add_argument(
        "--references-file",
        type=str,
        default=None,
        help="Text file: one reference per line (same order as prompts).",
    )
    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=128,
        help="Max number of *new* tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write metrics + examples as JSON.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of prompts for quick debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = None
    if args.eval_config is not None:
        cfg = load_yaml(args.eval_config)

    def get_cfg(key: str, cli_value, default=None):
        # wenn CLI Wert gesetzt -> CLI gewinnt
        if cli_value is not None:
            return cli_value
        if cfg is not None and key in cfg:
            return cfg[key]
        return default

    model_dir = Path(get_cfg("model_dir", args.model_dir))
    tokenizer_dir = Path(get_cfg("tokenizer_dir", args.tokenizer_dir))
    prompts_path = Path(get_cfg("prompts_file", args.prompts_file))
    refs_file = get_cfg("references_file", args.references_file)
    refs_path = Path(refs_file) if refs_file is not None else None

    max_gen_length = int(get_cfg("max_gen_length", args.max_gen_length, 128))
    device_name = str(get_cfg("device", args.device, "auto"))
    batch_size = int(get_cfg("batch_size", args.batch_size, 4))
    num_examples = get_cfg("num_examples", args.num_examples, None)
    out_json = get_cfg("out_json", args.out_json, None)

    device = choose_device(device_name)
    print(f"Using device: {device}")

    print(f"Loading model from {model_dir}")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)

    print(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    print(f"Loading prompts from {prompts_path}")
    prompts = load_lines(prompts_path)

    references: Optional[List[str]] = None
    if refs_path is not None:
        print(f"Loading references from {refs_path}")
        references = load_lines(refs_path)
        if len(references) != len(prompts):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) and references "
                f"({len(references)}) must match."
            )

    if num_examples is not None:
        prompts = prompts[: num_examples]
        if references is not None:
            references = references[: num_examples]

    print(f"Generating outputs for {len(prompts)} prompts...")
    generations = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_gen_length=max_gen_length,
        device=device,
        batch_size=batch_size,
    )

    # ----- Metrics -----
    metrics: Dict[str, Any] = {}

    if references is not None:
        bleu = compute_bleu(references, generations)
        if bleu is not None:
            metrics["bleu"] = bleu

        rouge = compute_rouge_l(references, generations)
        if rouge is not None:
            metrics.update(rouge)

        bert = compute_bertscore(references, generations)
        if bert is not None:
            metrics.update(bert)

    metrics.update(compute_repetition_stats(generations))
    metrics.update(length_stats(generations, references))

    print("\n=== Evaluation metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if out_json is not None:
        out_path = Path(out_json)
        out = {
            "metrics": metrics,
            "examples": [
                {
                    "prompt": p,
                    "generation": g,
                    "reference": r if references is not None else None,
                }
                for p, g, r in zip(
                    prompts,
                    generations,
                    references if references is not None else [None] * len(prompts),
                )
            ],
        }
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved detailed results to {out_path}")
