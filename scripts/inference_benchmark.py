"""
Benchmarking script to measure inference speed (Tokens/Sec) and Memory Usage.
Compares:
1. KV-Caching Enabled vs Disabled
2. VRAM usage

python scripts/benchmark.py --max_tokens 1000 --d_model 256
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import argparse
import pandas as pd
from pocket_narrator.models import get_model
from pocket_narrator.tokenizers import get_tokenizer

def measure_generation(model, tokenizer, prompt, max_new_tokens, use_cache, device):
    """
    Runs a single generation pass and measures metrics.
    """
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    input_tokens = [tokenizer.encode(prompt)]
    
    if device == "cuda": torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()
    
    start_time = time.time()
    
    _ = model.predict_sequence_batch(
        input_tokens, 
        max_length=max_new_tokens, 
        strategy="greedy", 
        use_cache=use_cache
    )
    
    if device == "cuda": torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()
    
    end_time = time.time()
    
    duration = end_time - start_time
    tokens_per_sec = max_new_tokens / duration
    
    memory_usage = 0.0
    if device == "cuda":
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2 # MB
    elif device == "mps":
        memory_usage = torch.mps.current_allocated_memory() / 1024**2 # MB estimation
        
    return tokens_per_sec, duration, memory_usage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--max_tokens", type=int, default=100, help="Tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Benchmarking on {device.upper()} ---")

    tokenizer = get_tokenizer("character")
    tokenizer.train(["a", "b", "c"])
    
    vocab_size = 128
    model_config = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_head": 4,
        "max_len": 1024,
        "activation_type": args.activation,
        "pos_encoding_type": "rope",
        "attention_type": "multi_head"
    }
    
    model = get_model("transformer", vocab_size=vocab_size, **model_config)
    model.to(device)
    
    print(f"Model Config: {args.d_model} dim, {args.n_layers} layers, {args.activation}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = []
    
    print("Warming up...")
    measure_generation(model, tokenizer, "a", 10, False, device)
    measure_generation(model, tokenizer, "a", 10, True, device)

    prompt = "a" * 10
    
    modes = [("Naive (No Cache)", False), ("Efficient (KV Cache)", True)]
    
    for name, use_cache in modes:
        print(f"\nRunning: {name}...")
        tps_list = []
        for i in range(5):
            tps, _, mem = measure_generation(model, tokenizer, prompt, args.max_tokens, use_cache, device)
            tps_list.append(tps)
            print(f"  Run {i+1}: {tps:.2f} tok/s")
        
        avg_tps = sum(tps_list) / len(tps_list)
        results.append({
            "Mode": name,
            "Avg Tokens/Sec": avg_tps,
            "Peak Memory (MB)": mem
        })

    df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(df.to_string(index=False))
    
    naive = df[df["Mode"] == "Naive (No Cache)"]["Avg Tokens/Sec"].values[0]
    cached = df[df["Mode"] == "Efficient (KV Cache)"]["Avg Tokens/Sec"].values[0]
    print(f"\nSpeedup Factor: {cached / naive:.2f}x")

if __name__ == "__main__":
    main()