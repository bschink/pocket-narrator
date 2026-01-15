# this mamba_evaluation with  :

"""
            "bleu": bleu.score
            "rougeL": rouge_l,
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast

# optional, when we  will install:
# pip install sacrebleu rouge-score bert-score
try:
    import sacrebleu
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
except ImportError:
    sacrebleu = None
    rouge_scorer = None
    bert_score = None

from .mamba_model import MambaLM
from .mamba_utils import HFDatasetWrapper


@dataclass
class EvalConfig:
    lm_dataset_dir: str
    batch_size: int = 32
    device: str = "cuda"


class MambaEvaluator:
    def __init__(self, model: MambaLM, tokenizer: PreTrainedTokenizerFast, cfg: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        ds = load_from_disk(cfg.lm_dataset_dir)
        if isinstance(ds, dict) and "validation" in ds:
            self.eval_ds = ds["validation"]
        else:
            self.eval_ds = ds

        self.loader = DataLoader(
            HFDatasetWrapper(self.eval_ds),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )

    @torch.no_grad()
    def perplexity(self, max_batches: Optional[int] = None) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(self.loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            n_tok = (batch["labels"] != self.model.config.pad_token_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        mean_loss = total_loss / max(total_tokens, 1)
        return float(torch.exp(torch.tensor(mean_loss)))

    @torch.no_grad()
    def bleu_rouge_bert(
        self,
        prompts: List[str],
        references: List[str],
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Convenience-Function: generiere für gegebene Prompts Output
        und vergleiche mit den Referenz-Texten.
        """
        if sacrebleu is None or rouge_scorer is None or bert_score is None:
            raise ImportError(
                "Please install 'sacrebleu', 'rouge-score' und 'bert-score', "
                "or this Methods not used."
            )

        self.model.eval()
        generations = []

        from .mamba_generate import generate_text

        for p in prompts:
            gen = generate_text(
                self.model,
                self.tokenizer,
                prompt=p,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=None,
                device=str(self.device),
            )
            generations.append(gen)

        # BLEU
        bleu = sacrebleu.corpus_bleu(generations, [references])

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l_scores = [
            scorer.score(ref, gen)["rougeL"].fmeasure for ref, gen in zip(references, generations)
        ]
        rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

        # BERTScore
        P, R, F1 = bert_score(generations, references, lang="en", verbose=False)

        return {
            "bleu": bleu.score,
            "rougeL": rouge_l,
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
