# gpt2.py  ---> our Model
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import GPT2LMHeadModel
import torch

from config_utils import load_yaml
from bpe_tokenizer_utils import train_bpe_tokenizer, load_bpe_tokenizer


class TextGenatate:
    def __init__(self, tokenizer_config_path: str):
        self.cfg = load_yaml(tokenizer_config_path)
        assert self.cfg["type"] == "bpe", "Tokenizer config must have type: bpe"

        ds_cfg = self.cfg["dataset"]
        self.block_size = int(self.cfg["block_size"])
        self.save_dir = Path(self.cfg["save_dir"])
        self.lm_dataset_dir = Path(self.cfg["lm_dataset_dir"])

        # 1) Download TinyStories from HF
        full_ds = load_dataset(
            ds_cfg["hf_dataset"],
            ds_cfg["config"],
            split=ds_cfg["split"],
        )
        start = ds_cfg.get("offset", 0)
        end = start + ds_cfg["num_rows"]
        sub_ds = full_ds.select(range(start, end))

        texts = sub_ds["text"]
        print("Sample from corpus:")
        print(texts[0][:500])

        # 2) Train or load BPE tokenizer
        tokenizer_path_exists = (self.save_dir / "tokenizer.json").exists()
        if tokenizer_path_exists:
            print(f"Loading existing BPE tokenizer from {self.save_dir}")
            self.tokenizer = load_bpe_tokenizer(self.save_dir)
        else:
            print(
                f"Training new BPE tokenizer (vocab_size={self.cfg['vocab_size']}) "
                f"and saving to {self.save_dir}"
            )
            self.tokenizer = train_bpe_tokenizer(
                texts=texts,
                vocab_size=self.cfg["vocab_size"],
                save_dir=self.save_dir,
            )

        # 3) (Optional) Dummy model for device check, will be rebuilt in the trainer.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Wrap HF Dataset for Tokenization
        self.row_data = Dataset.from_dict({"text": texts})

    # Tokenizer-Function
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.block_size,
        )

    def groupText(self, examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // self.block_size) * self.block_size

        result = {
            k: [
                concatenated[k][i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def main(self):
        tokenize_dataset = self.row_data.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
        )
        lm_dataset = tokenize_dataset.map(
            self.groupText,
            batched=True,
            num_proc=4,
        )

        self.lm_dataset_dir.mkdir(parents=True, exist_ok=True)
        lm_dataset.save_to_disk(str(self.lm_dataset_dir))
        print(f"Saved LM dataset to {self.lm_dataset_dir}")
