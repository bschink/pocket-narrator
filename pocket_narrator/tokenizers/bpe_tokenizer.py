"""
Contains the implementation of a Byte-Pair Encoding (BPE) tokenizer with Regex that learns
its vocabulary from a training corpus.
The implementation is close to the GPT-2/GPT-4 tokenizers and https://github.com/karpathy/minbpe
"""
import os
import regex as re
import unicodedata
import json
from .base_tokenizer import AbstractTokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BPETokenizer(AbstractTokenizer):
    """A BPE tokenizer with Regex that learns its vocabulary from data."""
    
    def __init__(self, vocab_size: int, pattern=GPT4_SPLIT_PATTERN):
        """
        - pattern: optional string to override the default
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 for BPE.")
        self.vocab_size = vocab_size
        self.merges = {} # (int, int) -> int
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern =  pattern
        self.compiled_pattern = re.compile(self.pattern)
    
    def train(self, corpus: list[str], verbose: bool = False):
        # join to single block of text
        text = "".join(corpus)
        num_merges = self.vocab_size - 256

        # split & preprocess text according to regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = self._get_stats(chunk_ids, stats)

            if not stats:
                print(f"INFO: No more pairs to merge after {i} merges. Stopping early.")
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.vocab = vocab
        self.merges = merges

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text: str) -> list[int]:
        return self._encode_internal(text, allowed_special="none_raise")

    def _encode_internal(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
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
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def _get_stats(self, ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def _encode_chunk(self, chunk_bytes):
        """
        Encode a single chunk of bytes using the learned BPE merges.
        
        Args:
            chunk_bytes: bytes object to encode
            
        Returns:
            list of token ids
        """
        ids = list(chunk_bytes)
        
        while len(ids) >= 2:
            stats = self._get_stats(ids, {})
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
            
        return ids
    
    def _encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def save(self, save_path: str):
        """
        Saves the tokenizer state to a specified directory.

        This will create two files inside the directory:
        - bpe.model: The core model file with merges and special tokens, for loading.
        - bpe.vocab: A human-readable vocabulary file for inspection.

        Args:
            save_path (str): The path to the DIRECTORY where files will be saved.
        """
        if not self.merges:
            raise ValueError("Cannot save an untrained BPE tokenizer.")
        
        print(f"INFO: Saving BPE tokenizer to directory: {save_path}")
        os.makedirs(save_path, exist_ok=True)

        model_file = os.path.join(save_path, "bpe.model")
        vocab_file = os.path.join(save_path, "bpe.vocab")

        with open(model_file, 'w', encoding="utf-8") as f:
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = self._render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self._render_token(self.vocab[idx0])
                    s1 = self._render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    @classmethod
    def load(cls, load_path: str):
        """
        Loads the tokenizer state from a directory.

        This expects to find a 'bpe.model' file inside the specified directory.

        Args:
            load_path (str): The path to the DIRECTORY containing the model file.
        
        Returns:
            An initialized and loaded BPETokenizer instance.
        """
        print(f"INFO: Loading BPE tokenizer from directory: {load_path}")
        model_file = os.path.join(load_path, "bpe.model")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Required model file 'bpe.model' not found in {load_path}")

        tokenizer = cls(vocab_size=256)

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            tokenizer.pattern = f.readline().strip()
            tokenizer.compiled_pattern = re.compile(tokenizer.pattern)
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        tokenizer.merges = merges
        tokenizer.special_tokens = special_tokens
        tokenizer.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        
        tokenizer.vocab = tokenizer._build_vocab()
        
        tokenizer.vocab_size = len(tokenizer.vocab)

        return tokenizer

    def _replace_control_characters(self, s: str) -> str:
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}") # escape
        return "".join(chars)

    def _render_token(self, t: bytes) -> str:
        # pretty print a token, escaping control characters
        s = t.decode('utf-8', errors='replace')
        s = self._replace_control_characters(s)
        return s