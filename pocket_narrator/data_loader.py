import os, random
from typing import List, Tuple, Iterator

def load_text_dataset(path: str) -> List[str]:
    """Load plain text where each non-empty line is a sample."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("Empty text dataset")
    return lines

def split_text(lines: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministic train/val split."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    idxs = list(range(len(lines)))
    random.Random(seed).shuffle(idxs)
    cut = int(len(lines) * (1 - val_ratio))
    train = [lines[i] for i in idxs[:cut]]
    val = [lines[i] for i in idxs[cut:]]
    return train, val

def batchify_text(lines: List[str], batch_size: int = 2, shuffle: bool = True, seed: int = 42) -> Iterator[List[str]]:
    """Yield batches of strings."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    idxs = list(range(len(lines)))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    for s in range(0, len(idxs), batch_size):
        yield [lines[i] for i in idxs[s:s + batch_size]]

