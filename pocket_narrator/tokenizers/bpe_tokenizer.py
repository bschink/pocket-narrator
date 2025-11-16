"""
Contains the implementation of a Byte-Pair Encoding (BPE) tokenizer with Regex.
This version is optimized for memory and computation to handle large datasets
by training on a stream of data.
"""
import os
import regex as re
import unicodedata
import json
from typing import Iterator
from tqdm import tqdm
from .base_tokenizer import AbstractTokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r{
\n}]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BPETokenizer(AbstractTokenizer):
    """A BPE tokenizer that learns its vocabulary from a training data stream."""
    
    def __init__(self, vocab_size: int, special_tokens: dict[str, int], pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 for BPE.")
        self.vocab_size = vocab_size
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.register_special_tokens(special_tokens)
        self.vocab = self._build_vocab()

    def train(self, corpus_iterator: Iterator[list[str]], verbose: bool = False):
        """
        Trains the BPE tokenizer from an iterator over a corpus. This approach is
        memory-efficient and suitable for large datasets.
        """
        if self.vocab_size <= 256 + len(self.special_tokens):
            return  # Nothing to learn
        num_merges = self.vocab_size - (256 + len(self.special_tokens))

        # initial stats
        print("INFO: (Phase 1/2) Building initial pair statistics from corpus stream...")
        stats = {}
        for batch in tqdm(corpus_iterator, desc="Processing corpus"):
            for text in batch:
                text_chunks = re.findall(self.compiled_pattern, text)
                for chunk in text_chunks:
                    ids = list(chunk.encode("utf-8"))
                    for pair in zip(ids, ids[1:]):
                        stats[pair] = stats.get(pair, 0) + 1

        # merges
        print("INFO: (Phase 2/2) Performing BPE merges...")
        merges = {}
        for i in tqdm(range(num_merges), desc="Merging pairs", unit="merge"):
            if not stats:
                print(f"INFO: No more pairs to merge after {i} merges. Stopping early.")
                break
        
            pair = max(stats, key=stats.get)
            
            # remove the merged pair and don't try to re-calculate all newly formed adjacent pairs, 
            # as that would require another pass over the full dataset
            stats.pop(pair)
            
            idx = 256 + i
            merges[pair] = idx

        self.merges = merges
        # rebuild the vocabulary with new merges
        self.vocab = self._build_vocab()
        
    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text: str) -> list[int]:
        return self._encode_internal(text, allowed_special="all")

    def _encode_internal(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            if self.special_tokens:
                assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self._encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids
    
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        return [self.decode(tokens) for tokens in token_lists]

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab.get(p0, b'') + vocab.get(p1, b'')
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def _get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids
    
    def _encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "bpe.model")

        with open(model_file, 'w', encoding="utf-8") as f:
            # encode as JSON to handle special characters
            f.write(f"{json.dumps(self.pattern)}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    @classmethod
    def load(cls, load_path: str):
        model_file = os.path.join(load_path, "bpe.model")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Required model file 'bpe.model' not found in {load_path}")

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # Decode from JSON to handle special characters
            pattern = json.loads(f.readline().strip())
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        # vocab size will be determined by the loaded merges and special tokens
        vocab_size = 256 + len(merges) + len(special_tokens)
        tokenizer = cls(vocab_size=vocab_size, special_tokens=special_tokens, pattern=pattern)
        tokenizer.merges = merges
        tokenizer.vocab = tokenizer._build_vocab()
        
        return tokenizer

    def _render_token(self, t: bytes) -> str:
        s = t.decode('utf-8', errors='replace')
        return "".join(f"\\u{ord(ch):04x}" if unicodedata.category(ch)[0] == "C" else ch for ch in s)