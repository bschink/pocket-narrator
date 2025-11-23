"""
HuggingFace TinyStories baseline model in order to the Edlan et al..

Wraps a pretrained TinyStories model from HF so that it implements
the AbstractLanguageModel interface used in pocket-narrator.
"""

from typing import List
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import AbstractLanguageModel


class HuggingFaceTinyStoriesLM(AbstractLanguageModel):
    """
    Wrapper around a pretrained TinyStories-* GPT-Neo model from HuggingFace.

    Example model names:
      - "roneneldan/TinyStories-1M"
      - "roneneldan/TinyStories-3M"
      - "roneneldan/TinyStories-9M"
      - "roneneldan/TinyStories-28M"
      - "roneneldan/TinyStories-33M"
    """

    def __init__(
        self,
        model_name: str = "roneneldan/TinyStories-28M",
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(vocab_size=0)  # we'll override vocab_size() anyway

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._vocab_size = len(self.tokenizer.get_vocab())
        self.eos_token_id = self.tokenizer.eos_token_id

    
    # AbstractLanguageModel API
   
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def next_token_logprobs(self, input_ids: List[int]) -> List[float]:
        """
        Return log-probabilities for the next token given the prefix.
        """
        x = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        logits = self.model(x).logits[:, -1, :]  # (1, vocab_size)
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[int]:
        """
        Simple autoregressive sampling with temperature and nucleus (top-p) sampling.
        """
        ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

        for _ in range(max_new_tokens):
            logits = self.model(ids).logits[:, -1, :]  # (1, vocab_size)
            logits = logits / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                # Nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                # Mask tokens outside nucleus
                mask = cum_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_id = torch.multinomial(sorted_probs, num_samples=1)  # (1,1)
                next_id = sorted_idx.gather(-1, next_id)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            if self.eos_token_id is not None and next_id.item() == self.eos_token_id:
                break

        return ids[0].tolist()
