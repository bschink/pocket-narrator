#Train a BPE tokenizer on TinyStories and create LM dataset (chunks of length block_size).

import argparse
from pathlib import Path
import pandas as pd

from datasets import load_dataset, DatasetDict, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from pocket_narrator.models.mamba.config_utils import load_yaml
"""
python -m pocket_narrator.models.mamba.train_tokenizer_and_lm_dataset \
  --config configs/train_tokenizer_and_lm_dataset.yaml

  """

def _load_ds(dataset_cfg: dict, split: str):
    """
    Load dataset from either:
    - local text file (datasets 'text' loader) if dataset_cfg['local_text_file'] is set
    - HuggingFace Hub otherwise (reuse cache if it exists)
    """
    local_file = dataset_cfg.get("local_text_file", None)
    if local_file:
        # 'text' dataset returns a column named 'text'
        return load_dataset("text", data_files=local_file, split="train")

    hf_dataset = dataset_cfg["hf_dataset"]
    config_name = dataset_cfg.get("config", None)

    return load_dataset(
        hf_dataset,
        config_name,
        split=split,
        download_mode="reuse_dataset_if_exists",
    )

def build_tokenizer(c: dict):
    dataset_cfg = c["dataset"]
    tok_cfg = c["tokenizer"]

    hf_dataset = dataset_cfg["hf_dataset"]
    split = dataset_cfg.get("split", "train")
    config_name = dataset_cfg.get("config", None)
    num_rows = dataset_cfg.get("num_rows", None)
    text_key = dataset_cfg.get("text_key", "text")

    save_dir = Path(tok_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_txt = save_dir / "corpus.txt"

    ds = _load_ds(dataset_cfg, split=split)
    if num_rows is not None:
        ds = ds.select(range(min(num_rows, len(ds))))

    with tmp_txt.open("w", encoding="utf-8") as f:
        for ex in ds:
            text = ex[text_key]
            f.write(text.replace("\n", " ") + "\n")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=tok_cfg["vocab_size"],
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    )
    tokenizer.train([str(tmp_txt)], trainer)
    block_size = tok_cfg.get("block_size", 256)
    tokenizer.enable_truncation(max_length=block_size)


    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tok.save_pretrained(str(save_dir))
    print(f"[Tokenizer] Saved to {save_dir}")

    return hf_tok


def build_lm_dataset(c: dict, tokenizer: PreTrainedTokenizerFast):
    dataset_cfg = c["dataset"]
    tok_cfg = c["tokenizer"]

    hf_dataset = dataset_cfg["hf_dataset"]
    split = dataset_cfg.get("split", "train")
    config_name = dataset_cfg.get("config", None)
    num_rows = dataset_cfg.get("num_rows", None)
    text_key = dataset_cfg.get("text_key", "text")

    block_size = tok_cfg.get("block_size", 256)
    lm_dir = Path(tok_cfg["lm_dataset_dir"])
    lm_dir.mkdir(parents=True, exist_ok=True)

    ds = _load_ds(dataset_cfg, split=split)
    print('------------------')
    print(type(ds))
    df= pd.DataFrame(ds)
    print(df.info)
    print(f"data shape : {ds.data.shape}")
    print('------------------')
    if num_rows is not None:
        ds = ds.select(range(min(num_rows, len(ds))))

    def tokenize_fn(example):
        ids = tokenizer(
            example[text_key],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        total = (len(ids) // block_size) * block_size
        ids = ids[:total]

        chunks = [ids[i : i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": chunks}

    tokenized = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
    )

    all_ids = []
    for ex in tokenized:
        all_ids.extend(ex["input_ids"])

    train_ds = Dataset.from_dict({"input_ids": all_ids})
    new_ds = DatasetDict({"train": train_ds})

    new_ds.save_to_disk(str(lm_dir))
    print(f"[LM dataset] Saved to {lm_dir}")
    print(f"[LM dataset] Num sequences: {len(train_ds)} | block_size={block_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print("[DEBUG] Loaded cfg keys:", cfg.keys())
    print("[DEBUG] tokenizer cfg:", cfg.get("tokenizer", None))

    tok = build_tokenizer(cfg)
    build_lm_dataset(cfg, tok)


if __name__ == "__main__":
    main()


