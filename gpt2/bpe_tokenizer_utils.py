# bpe_tokenizer_utils.py
from pathlib import Path
from typing import Iterable, List, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast



SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    save_dir: Union[str, Path],
) -> PreTrainedTokenizerFast:
    """
    Train a byte-pair tokenizer from raw texts and wrap it as HF tokenizer.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )

    def _iterator():
        for t in texts:
            if isinstance(t, str):
                yield t
            else:
                yield str(t)

    tok.train_from_iterator(_iterator(), trainer=trainer)

    # save raw tokenizer.json
    tok.save(str(save_dir / "tokenizer.json"))

    # Wrap as HF fast tokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )

    hf_tok.save_pretrained(save_dir)
    return hf_tok


def load_bpe_tokenizer(save_dir: Union[str, Path]) -> PreTrainedTokenizerFast:
    """
    Load the previously saved HF fast tokenizer.
    """
    save_dir = Path(save_dir)
    return PreTrainedTokenizerFast.from_pretrained(save_dir)
