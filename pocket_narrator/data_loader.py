import os, random
from typing import List, Tuple, Iterator
from typing import Iterable, Set, Optional, Callable


# ----- Basic text dataset utilities for MVP -----

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


# ---------------------------------------------------------------------------
# Streaming utilities for huge .txt files (e.g., TinyStories 1.8GB)
# ---------------------------------------------------------------------------


def stream_documents_from_txt(
    path: str,
    delimiter: str = "<|endoftext|>",
    normalize_ws: bool = True,
    drop_empty: bool = True,
) -> Iterable[str]:
    """
    Stream documents from a giant .txt file, splitting on a delimiter token.
    Yields one document at a time, never loading the whole file in memory.

    Example input:
      ... story A ... <|endoftext|>
      ... story B ... <|endoftext|>

    Example output:
      "…story A…"
      "…story B…"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")

    buf = []
    delim = delimiter.strip()

    # errors='ignore' prevents rare decoding crashes on odd bytes
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if delim in line:
                before, _, after = line.partition(delim)
                if before:
                    buf.append(before)
                doc = "".join(buf)
                if normalize_ws:
                    doc = _normalize_whitespace(doc)
                if not (drop_empty and _is_empty(doc)):
                    yield doc
                buf = []
                if after and after.strip():
                    buf.append(after)
            else:
                buf.append(line)

    # flush tail if file didn't end with the delimiter
    if buf:
        doc = "".join(buf)
        if normalize_ws:
            doc = _normalize_whitespace(doc)
        if not (drop_empty and _is_empty(doc)):
            yield doc


def _normalize_whitespace(text: str) -> str:
    """
    Keep normal spaces/newlines, but:
    - strip leading/trailing whitespace
    - strip trailing spaces on each line
    - collapse runs of blank lines to at most two
    """
    text = text.strip()
    if not text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out)


def _is_empty(text: str) -> bool:
    return len(text.strip()) == 0


def scan_unique_characters(path: str, delimiter: str = "<|endoftext|>") -> Set[str]:
    """
    Single-pass character scan. Returns a set of all distinct characters
    found across all documents, streaming safely for huge files.
    """
    chars: Set[str] = set()
    # Note: normalize_ws=False to include original characters as-is
    for doc in stream_documents_from_txt(path, delimiter=delimiter, normalize_ws=False, drop_empty=True):
        chars.update(doc)
    return chars


def preprocess_txt_to_bos_eos(
    input_path: str,
    output_path: str,
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    delimiter: str = "<|endoftext|>",
    normalize_ws: bool = True,
    drop_empty: bool = True,
    progress_log: Optional[Callable[[int], None]] = None,
) -> None:
    """
    Convert a giant .txt corpus (split by delimiter) into a new file where
    each document is wrapped with <bos> and <eos>, sequentially:

      <bos>
      ...doc text...
      <eos>
      <bos>
      ...doc text...
      <eos>

    Writes line-by-line to keep memory use tiny.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc in stream_documents_from_txt(
            input_path,
            delimiter=delimiter,
            normalize_ws=normalize_ws,
            drop_empty=drop_empty,
        ):
            out_f.write(f"{bos_token}\n")
            out_f.write(doc)
            out_f.write("\n")
            out_f.write(f"{eos_token}\n")
            count += 1
            if progress_log and count % 1000 == 0:
                progress_log(count)

