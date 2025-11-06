"""
Interpolated Kneser–Ney N-gram language model (from scratch)

This module implements a self-contained, training-ready N-gram LM that
conforms to the LMWrapper interface used by evaluate.py:

class LMWrapper(Protocol):
    def vocab_size(self) -> int: ...
    def next_token_logprobs(self, input_ids: List[int]) -> List[float]: ...
    def generate(self, prompt_ids: List[int], *, max_new_tokens:int, temperature:float=1.0, top_p:float=1.0) -> List[int]: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...

Usage example
-------------
from pocket_narrator.ngram import NGramLM

texts = [
    "Once upon a time, Sam had a red ball.",
    "Once upon a time, Ana met Sam in the park.",
]
lm = NGramLM(n=4, discount=0.75, min_count=1)
lm.fit(texts)

prompt = "Once upon a time,"
out_ids = lm.generate(lm.encode(prompt), max_new_tokens=50, temperature=0.8, top_p=0.9)
print(lm.decode(out_ids))

Notes
-----
- Word-level tokenization with lightweight regex, punctuation is kept as tokens.
- Interpolated Kneser–Ney smoothing (constant discount D).
- <bos> and <eos> handle boundaries; <unk> for OOVs.
- Nucleus (top-p) sampling with temperature for generation.
- Pure Python, no external deps.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# Tokenizer (simple, deterministic)

_WORD_RE = re.compile(r"[A-Za-z']+|[.,!?;:]")


def _tokenize(text: str) -> List[str]:
    toks = _WORD_RE.findall(text)
    # lowercase words, keep punctuation as is
    out: List[str] = []
    for t in toks:
        if t.isalpha() or (t.replace("'", "").isalpha() and "'" in t):
            out.append(t.lower())
        else:
            out.append(t)
    return out


def _detok(tokens: Sequence[str]) -> str:
    # join tokens with spaces, but remove space before punctuation
    s = " ".join(tokens)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    # capitalize first letter of sentences lightly
    def cap_after(match: re.Match) -> str:
        return match.group(0)[:-1] + match.group(0)[-1].upper()
    s = re.sub(r"(^\s*[a-z])", lambda m: m.group(0).upper(), s)
    s = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    return s



# Core Kneser–Ney N-gram model

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def build(cls, tokens: List[str]) -> "Vocab":
        uniq = [BOS, EOS, UNK]
        seen = set(uniq)
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        stoi = {t: i for i, t in enumerate(uniq)}
        itos = {i: t for t, i in stoi.items()}
        return cls(stoi, itos)


class NGramLM:
    def __init__(self, n: int = 4, discount: float = 0.75, min_count: int = 1, seed: int = 1337):
        assert n >= 1
        self.n = n
        self.D = discount
        self.min_count = min_count
        self.rng = random.Random(seed)

        self.vocab: Vocab | None = None
        # n-gram counts for orders 1..n
        self.ng_counts: List[Counter] = []
        # history (context) counts for orders 0..n-1 (0th unused)
        self.hist_counts: List[Counter] = []
        # number of distinct successors for each history (for lambda)
        self.succ_types: List[Dict[Tuple[int, ...], int]] = []
        # unigram continuation counts (distinct left contexts)
        self.cont_counts: Counter = Counter()
        self.total_continuations: int = 0

    # -------- Tokenization & I/O ----------
    def encode(self, text: str) -> List[int]:
        assert self.vocab is not None, "Model not fitted. Call fit() first."
        toks = [BOS] * (self.n - 1) + _tokenize(text) + [EOS]
        return [self.vocab.stoi.get(t, self.vocab.stoi[UNK]) for t in toks]

    def decode(self, ids: List[int]) -> str:
        assert self.vocab is not None, "Model not fitted. Call fit() first."
        toks = [self.vocab.itos.get(i, UNK) for i in ids]
        # strip BOS/EOS
        toks = [t for t in toks if t not in (BOS,)]
        if toks and toks[-1] == EOS:
            toks = toks[:-1]
        return _detok(toks)

    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab.stoi)

    # -------- Training ----------
    def _prepare_sequences(self, texts: Sequence[str]) -> List[List[str]]:
        seqs: List[List[str]] = []
        for txt in texts:
            toks = _tokenize(txt)
            seq = [BOS] * (self.n - 1) + toks + [EOS]
            seqs.append(seq)
        return seqs

    def _collect_tokens(self, seqs: List[List[str]]) -> List[str]:
        flat: List[str] = []
        for s in seqs:
            flat.extend(t for t in s if t not in (BOS, EOS))
        return flat

    def _trim_rare(self, seqs: List[List[str]]) -> List[List[str]]:
        if self.min_count <= 1:
            return seqs
        counts = Counter(t for s in seqs for t in s)
        keep = {t for t, c in counts.items() if c >= self.min_count} | {BOS, EOS, UNK}
        out: List[List[str]] = []
        for s in seqs:
            out.append([t if t in keep else UNK for t in s])
        return out

    def fit(self, texts: Sequence[str]) -> None:
        seqs = self._prepare_sequences(texts)
        # Build vocab on raw tokens (then maybe trim)
        all_tokens = self._collect_tokens(seqs)
        self.vocab = Vocab.build(all_tokens)

        # Replace rare tokens with UNK if needed
        seqs = self._trim_rare(seqs)

        # Convert to ids
        id_seqs = [[self.vocab.stoi.get(t, self.vocab.stoi[UNK]) for t in s] for s in seqs]

        # Initialize containers
        self.ng_counts = [Counter() for _ in range(self.n + 1)]  # 0 unused
        self.hist_counts = [Counter() for _ in range(self.n)]    # index k holds (k-gram histories)
        succ_sets: List[defaultdict] = [defaultdict(set) for _ in range(self.n + 1)]
        cont_histories: defaultdict[int, set] = defaultdict(set)

        # Count n-grams for all orders
        for ids in id_seqs:
            L = len(ids)
            for i in range(L):
                for order in range(1, self.n + 1):
                    if i - order + 1 < 0:
                        break
                    ng = tuple(ids[i - order + 1:i + 1])
                    self.ng_counts[order][ng] += 1
                    if order >= 2:
                        hist = ng[:-1]
                        self.hist_counts[order - 1][hist] += 1
                        succ_sets[order][hist].add(ng[-1])
                        if order == 2:
                            cont_histories[ng[-1]].add(ng[0])

        # Successor type counts for lambda(history)
        self.succ_types = [defaultdict(int) for _ in range(self.n + 1)]
        for order in range(2, self.n + 1):
            for hist, s in succ_sets[order].items():
                self.succ_types[order][hist] = len(s)

        # Continuation counts for unigram Kneser–Ney base distribution
        self.cont_counts = Counter({w: len(hs) for w, hs in cont_histories.items()})
        self.total_continuations = sum(self.cont_counts.values())

    # -------- Kneser–Ney probabilities ----------
    def _p_continuation(self, wid: int) -> float:
        # Unigram continuation prob P_KN(w)
        c = self.cont_counts.get(wid, 0)
        if self.total_continuations == 0:
            # uniform fallback
            return 1.0 / self.vocab_size()
        return c / self.total_continuations

    def _lambda(self, order: int, history: Tuple[int, ...]) -> float:
        # Backoff weight for a given history
        if order == 1:
            return 0.0
        hist_count = self.hist_counts[order - 1].get(history, 0)
        if hist_count == 0:
            return 1.0  # fully back off
        ntypes = self.succ_types[order].get(history, 0)
        return (self.D * ntypes) / hist_count

    def _p_kn(self, history: Tuple[int, ...], wid: int, order: int) -> float:
        # Recursive interpolated Kneser–Ney probability P(w|history)
        if order == 1:
            return self._p_continuation(wid)
        # count(history,w)
        count_hw = self.ng_counts[order].get(history + (wid,), 0)
        hist_count = self.hist_counts[order - 1].get(history, 0)
        first_term = 0.0
        if hist_count > 0:
            first_term = max(count_hw - self.D, 0.0) / hist_count
        backoff = self._lambda(order, history) * self._p_kn(history[1:], wid, order - 1)
        return first_term + backoff

    # -------- Public API: next_token_logprobs & generation ----------
    def next_token_logprobs(self, input_ids: List[int]) -> List[float]:
        assert self.vocab is not None, "Model not fitted. Call fit() first."
        # Take last n-1 tokens as history; if not enough, left-pad with BOS id
        bos_id = self.vocab.stoi[BOS]
        hist = tuple(([bos_id] * (self.n - 1) + input_ids)[- (self.n - 1):])
        # Compute probability for every vocab item
        V = self.vocab_size()
        lps = [0.0] * V
        for wid in range(V):
            p = self._p_kn(hist, wid, self.n)
            if p <= 0.0:
                lps[wid] = -1e9
            else:
                lps[wid] = math.log(p)
        return lps

    def _sample_from_logprobs(self, logps: List[float], temperature: float = 1.0, top_p: float = 1.0) -> int:
        # Temperature scaling
        if temperature <= 0:
            # argmax
            return max(range(len(logps)), key=lambda i: logps[i])
        scaled = [lp / max(1e-8, temperature) for lp in logps]
        # Convert to probs
        m = max(scaled)
        probs = [math.exp(lp - m) for lp in scaled]
        s = sum(probs)
        if s == 0:
            probs = [1.0 / len(probs)] * len(probs)
        else:
            probs = [p / s for p in probs]

        # Nucleus (top-p)
        if 0 < top_p < 1.0:
            idx = list(range(len(probs)))
            idx.sort(key=lambda i: probs[i], reverse=True)
            cum = 0.0
            keep = []
            for i in idx:
                keep.append(i)
                cum += probs[i]
                if cum >= top_p:
                    break
            mass = sum(probs[i] for i in keep)
            probs = [probs[i] / mass if i in keep else 0.0 for i in range(len(probs))]

        # Sample
        r = self.rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return len(probs) - 1

    def generate(self, prompt_ids: List[int], *, max_new_tokens: int, temperature: float = 1.0, top_p: float = 1.0) -> List[int]:
        out = list(prompt_ids)
        eos_id = self.vocab.stoi[EOS]
        for _ in range(max_new_tokens):
            logps = self.next_token_logprobs(out)
            wid = self._sample_from_logprobs(logps, temperature=temperature, top_p=top_p)
            out.append(wid)
            if wid == eos_id:
                break
        return out



# Minimal  test

if __name__ == "__main__":
    corpus = [
        "Once upon a time, Sam had a red ball.",
        "Once upon a time, Ana met Sam in the park.",
        "Sam liked the park.",
        "Ana liked the red ball.",
    ]
    lm = NGramLM(n=4, discount=0.75)
    lm.fit(corpus)

    print("Vocab size:", lm.vocab_size())

    prompt = "Once upon a time,"
    ids = lm.encode(prompt)
    gen = lm.generate(ids, max_new_tokens=30, temperature=0.8, top_p=0.9)
    print(lm.decode(gen))
