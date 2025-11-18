# wrapper for a gpt2 model from huggingface
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import AbstractLanguageModel

class GPT2Wrapper(AbstractLanguageModel):
    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        # we don't really use vocab_size in the same way, but base class expects it
        self.model_name = model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.hf_model.to(self.device)

        super().__init__(vocab_size=self.hf_model.config.vocab_size)

    def predict_sequence_batch(
        self,
        input_tokens_batch: List[List[int]],
        max_length: int = 50,
        strategy: str = "greedy",
        no_repeat_ngram_size: int | None = None,
    ) -> List[List[int]]:
        """
        For GPT-2, it's much easier to work in *text* space,
        so here we assume the IDs are HF token IDs.
        BUT in this project we work with our own tokenizers,
        so we'll instead accept *text prompts* later through a helper.
        For now we keep the signature to satisfy the interface.
        """
        raise NotImplementedError(
            "For GPT2Wrapper, use predict_from_prompts_text(...) instead of raw token IDs."
        )

    def predict_from_prompts_text(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> list[str]:
        """
        This is the practical method to call for comparison: give prompts, get continuations.
        """
        self.hf_model.eval()
        outputs = []

        for prompt in prompts:
            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)

            if strategy == "sample":
                do_sample = True
                top_k_val = top_k
                temperature_val = temperature
            else:
                do_sample = False
                top_k_val = 0
                temperature_val = 1.0

            with torch.no_grad():
                generated = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_k=top_k_val if do_sample else None,
                    temperature=temperature_val if do_sample else None,
                )

            text = self.hf_tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
            )
            # return only the new continuation (remove prompt prefix)
            if text.startswith(prompt):
                continuation = text[len(prompt):]
            else:
                continuation = text
            outputs.append(continuation)

        return outputs

    def save(self, model_path: str):
        """
        we normally don't retrain GPT-2 here, so saving is optional.
        We'll just save the model name in case we want it.
        """
        # no-op or small JSON write if you want
        pass

    @classmethod
    def load(cls, model_path: str, config: dict):
        """
        For a purely pre-trained GPT-2, we don't really "load" from a local file;
        we just 're-download' from Hugging Face.
        """
        return cls(model_name=config.get("model_name", "gpt2"))
