import argparse
import os
import yaml

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_bpe_tokenizer(cfg: dict) -> str:
    """
    Trainiert einen BPE-Tokenizer und speichert ihn als tokenizer.json im save_dir.
    Erwartet im YAML:
      - save_dir
      - vocab_size
      - special_tokens
      - dataset: { hf_dataset, config?, split, num_rows?, offset? }
    """
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")

    tok = Tokenizer(models.BPE(unk_token=cfg["special_tokens"][3]))  # "<unk>"
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=cfg["vocab_size"],
        special_tokens=cfg["special_tokens"],
        show_progress=True,
    )

    ds_cfg = cfg["dataset"]
    dataset = load_dataset(
        ds_cfg["hf_dataset"],
        ds_cfg.get("config", None),
        split=ds_cfg.get("split", "train"),
    )

    num_rows = ds_cfg.get("num_rows", None)
    offset = ds_cfg.get("offset", 0)

    if num_rows is not None:
        dataset = dataset.select(range(offset, offset + num_rows))

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            samples = dataset[i : i + batch_size]["text"]
            yield samples

    tok.train_from_iterator(batch_iterator(), trainer=trainer)

    bos = cfg["special_tokens"][1]
    eos = cfg["special_tokens"][2]
    tok.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A {eos}",
        pair=f"{bos} $A {eos} {bos} $B {eos}",
        special_tokens=[
            (bos, tok.token_to_id(bos)),
            (eos, tok.token_to_id(eos)),
        ],
    )

    tok.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer_path


def build_lm_dataset(cfg: dict, tokenizer_path: str):
    """
    Baut ein LM-Dataset mit block_size und speichert es in lm_dataset_dir.
    Erwartet im YAML:
      - block_size
      - lm_dataset_dir
      - dataset (wie oben)
      - special_tokens
    """
    block_size = cfg["block_size"]
    lm_dataset_dir = cfg["lm_dataset_dir"]
    os.makedirs(lm_dataset_dir, exist_ok=True)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token=cfg["special_tokens"][1],
        eos_token=cfg["special_tokens"][2],
        unk_token=cfg["special_tokens"][3],
        pad_token=cfg["special_tokens"][0],
    )
    fast_tok.model_max_length = block_size

    ds_cfg = cfg["dataset"]
    raw_datasets = load_dataset(
        ds_cfg["hf_dataset"],
        ds_cfg.get("config", None),
    )

    if "train" in raw_datasets:
        train_ds = raw_datasets["train"]
    else:
        first_split = list(raw_datasets.keys())[0]
        train_ds = raw_datasets[first_split]

    num_rows = ds_cfg.get("num_rows", None)
    offset = ds_cfg.get("offset", 0)
    if num_rows is not None:
        train_ds = train_ds.select(range(offset, offset + num_rows))

    def tokenize_function(examples):
        return fast_tok(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )

    tokenized = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
    )

    tokenized = tokenized.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True,
        desc="Creating labels",
    )

    tokenized.save_to_disk(lm_dataset_dir)
    print(f"LM dataset saved to {lm_dataset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Pfad zur Tokenizer-YAML-Config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    tokenizer_path = train_bpe_tokenizer(cfg)
    build_lm_dataset(cfg, tokenizer_path)


if __name__ == "__main__":
    main()
