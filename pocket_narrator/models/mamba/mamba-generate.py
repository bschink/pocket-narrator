# mamba_generate.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from .mamba_model import MambaLM


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    logits: (B, V) -> sampled token ids (B,)
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, idx = torch.topk(logits, top_k)
        probs = F.softmax(v, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
        return torch.gather(idx, -1, next_idx).squeeze(-1)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_text(
    model: MambaLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cuda",
) -> str:
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        logits = model.generate_step(input_ids)     # (B, V)
        next_token = sample_next_token(logits, temperature, top_k)  # (B,)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        # optional: stoppen bei EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
