"""
Strict-English Dataset Preprocessing Script.

- Splits raw .txt into stories using default <|endoftext|>.
- Removes:
    * stories with < min_letters ASCII letters
    * stories containing ANY non-ASCII characters (Chinese, umlauts, accents...)
    * stories containing ANY non-English words (based on wordfreq)
    * duplicates (exact text match)
- Formats output as:
        STORY_TEXT <|endoftext|>
- Automatically creates:
        <input>.clean.txt
        <input>.clean.half.txt
        <input>.clean.quarted.txt
        <input>.clean.deleted.txt
        <input>.clean.manifest.json
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import re
import random
from tqdm import tqdm
from wordfreq import zipf_frequency

# strict ASCII punctuation + allowed some marks 
ALLOWED_CHARS_REGEX = re.compile(
    r"[A-Za-z0-9\s.,;:'\"?!()\-\[\]/’‘“”…—`$“+’]"
)

# ------------------------------------------
# Helpers
# ------------------------------------------

def split_raw_file(input_path: str, delimiter: str) -> list[str]:
    """Split the raw dataset into stories."""
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split(delimiter)
    return [p.strip() for p in parts if p.strip()]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def detect_invalid_story(
    text: str,
    min_letters: int,
) -> list[str]:
    """
    Strict-English invalid story detection:

    ✔ Reject ANY non-ASCII alphabetic characters (Chinese, Cyrillic, äöü, é, etc.)
    ✔ Reject ANY non-allowed punctuation not in ALLOWED_CHARS_REGEX
    ✔ Require minimum ASCII alphabetic count
    """
    reasons = []

    # Count ASCII alphabetic letters only
    ascii_letter_count = sum(ch.isascii() and ch.isalpha() for ch in text)
    if ascii_letter_count < min_letters:
        reasons.append(f"too_few_letters({ascii_letter_count})")

    # Reject ANY non-ASCII characters immediately
    for ch in text:
        if ch.isspace():
            continue
        if not ch.isascii():
            reasons.append(f"invalid_chars(non_ascii:{repr(ch)})")
            return reasons
        if not ALLOWED_CHARS_REGEX.fullmatch(ch):
            reasons.append(f"invalid_chars({repr(ch)})")
            return reasons

    return reasons


def is_english_word(word: str, threshold: float) -> bool:
    """Use wordfreq zipf frequency to test English-ness."""
    if not word:
        return False
    return zipf_frequency(word.lower(), "en") >= threshold


def detect_non_english_strict(
    text: str,
    wordfreq_threshold: float,
) -> list[str]:
    """
    STRICT rule:

    ❗ If ANY token is not English → story removed.
    No ratio allowed.
    No tolerance.

    Steps:
      - Extract ONLY ASCII alphabetic tokens
      - Use wordfreq to classify English words
      - Reject if ANY fail
    """
    # Extract ASCII words only
    tokens = re.findall(r"[A-Za-z]+", text)

    if not tokens:
        return ["non_english(no_ascii_tokens)"]

    bad = [t for t in tokens if not is_english_word(t, wordfreq_threshold)]

    if bad:
        sample = ", ".join(bad[:5])
        return [f"non_english_words({sample})"]

    return []


def build_duplicate_map(stories: list[str]) -> dict[int, list[str]]:
    """Detect exact-text duplicates."""
    seen = {}
    dup_reasons = {}
    for idx, s in enumerate(stories):
        key = " ".join(s.split()).lower()
        if key in seen:
            dup_reasons[idx] = [f"duplicate_of_{seen[key]}"]
        else:
            seen[key] = idx
    return dup_reasons


def format_story_for_output(
    text: str,
    output_delimiter: str,
) -> str:
    """Format as: STORY <|endoftext|>"""
    return f"{text} {output_delimiter}"


# ------------------------------------------
# Main
# ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Strict-English Preprocessor")

    ap.add_argument("--input", required=True,
                    help="Path to raw .txt file")

    ap.add_argument("--output", required=True,
                    help="Destination folder or file")

    # input delimiter
    ap.add_argument("--delimiter", default="<|endoftext|>",
                    help="Raw-story delimiter")

    # whitespace
    ap.add_argument("--no-normalize-ws", action="store_true",
                    help="Disable whitespace normalization")

    # minimum letters
    ap.add_argument("--min-letters", type=int, default=100,
                    help="Minimum ASCII letters per story")

    # strict English settings
    ap.add_argument("--wordfreq-threshold", type=float, default=2.0,
                    help="Minimum zipf freq for English words (strict)") # scale is 0-7

    # keep or drop invalid
    ap.add_argument("--keep-invalid", action="store_true",
                    help="Keep invalid stories (do not delete)")

    # splitting
    ap.add_argument("--no-splits", action="store_true",
                    help="Disable half/quarter splits")

    ap.add_argument("--split-seed", type=int, default=42)

    args = ap.parse_args()

    # ------------------------------------------
    # Naming
    # ------------------------------------------

    input_stem = Path(args.input).stem  # e.g. TinyStoriesGPT4
    base_name = input_stem + ".clean"

    out_root = Path(args.output)
    if out_root.exists() and out_root.is_dir():
        out_dir = out_root
    elif str(args.output).endswith("/"):
        out_dir = out_root
    else:
        out_dir = out_root.parent

    clean_path = out_dir / f"{base_name}.txt"
    half_path = out_dir / f"{base_name}.half.txt"
    quarter_path = out_dir / f"{base_name}.quarted.txt"
    deleted_log_path = out_dir / f"{base_name}.deleted.txt"
    manifest_path = out_dir / f"{base_name}.manifest.json"

    drop_invalid = not args.keep_invalid
    make_splits = not args.no_splits

    # ------------------------------------------
    # Step 1 — load & split
    # ------------------------------------------

    print(f"[step 1] Loading {args.input}")
    stories = split_raw_file(args.input, args.delimiter)
    print(f"[step 1] Found {len(stories)} raw stories.")

    if not args.no_normalize_ws:
        print("[step 1] Normalizing whitespace…")
        stories = [normalize_whitespace(s) for s in tqdm(stories)]

    # ------------------------------------------
    # Step 2 — strict invalid checks
    # ------------------------------------------

    print("[step 2] Checking strict invalid chars + length")
    invalid_reasons = {}
    for idx, s in enumerate(tqdm(stories)):
        r = detect_invalid_story(s, args.min_letters)
        if r:
            invalid_reasons[idx] = r

    print("[step 2b] Checking duplicates")
    dup_reasons = build_duplicate_map(stories)

    print("[step 2c] Checking strict non-English (ANY bad word removes story)")
    non_english_reasons = {}
    for idx, s in enumerate(tqdm(stories)):
        r = detect_non_english_strict(s, args.wordfreq_threshold)
        if r:
            non_english_reasons[idx] = r

    # merge reasons
    all_reasons = {}
    for idx in range(len(stories)):
        temp = []
        if idx in invalid_reasons:
            temp.extend(invalid_reasons[idx])
        if idx in dup_reasons:
            temp.extend(dup_reasons[idx])
        if idx in non_english_reasons:
            temp.extend(non_english_reasons[idx])
        if temp:
            all_reasons[idx] = temp

    print(f"[step 2] Total stories with any issue: {len(all_reasons)}")

    # ------------------------------------------
    # Step 3 — Delete or keep
    # ------------------------------------------

    kept = []
    deleted = []

    for idx, s in enumerate(stories):
        r = all_reasons.get(idx, [])
        if r and drop_invalid:
            deleted.append((idx, s, r))
        else:
            kept.append(s)

    print(f"[step 3] Kept {len(kept)} stories. Deleted {len(deleted)}.")

    # write deleted log
    if drop_invalid:
        deleted_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[step 3] Writing deleted log → {deleted_log_path}")
        with open(deleted_log_path, "w", encoding="utf-8") as f:
            for idx, text, reasons in deleted:
                f.write(f"### STORY {idx} | reasons: {', '.join(reasons)}\n{text}\n\n")

    # ------------------------------------------
    # Step 4 — Format output
    # ------------------------------------------

    print(f"[step 4] Writing cleaned output → {clean_path}")
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    output_delim = "<|endoftext|>"
    with open(clean_path, "w", encoding="utf-8") as f:
        for s in tqdm(kept):
            f.write(format_story_for_output(s, output_delim) + "\n")

    # ------------------------------------------
    # Step 5 — Splits
    # ------------------------------------------

    half_file = None
    quarter_file = None

    if make_splits:
        print("[step 5] Creating 1/2 and 1/4 splits…")
        rng = random.Random(args.split_seed)
        indices = list(range(len(kept)))
        rng.shuffle(indices)

        n = len(kept)
        half_n = n // 2
        quarter_n = n // 4

        # Write half
        half_file = str(half_path)
        with open(half_file, "w", encoding="utf-8") as f:
            for i in indices[:half_n]:
                f.write(format_story_for_output(kept[i], output_delim) + "\n")

        # Write quarter
        quarter_file = str(quarter_path)
        with open(quarter_file, "w", encoding="utf-8") as f:
            for i in indices[:quarter_n]:
                f.write(format_story_for_output(kept[i], output_delim) + "\n")

        print(f"[step 5] half → {half_file}")
        print(f"[step 5] quarter → {quarter_file}")

    # ------------------------------------------
    # Step 6 — Manifest
    # ------------------------------------------

    print(f"[step 6] Writing manifest → {manifest_path}")

    manifest = {
        "input": args.input,
        "cleaned": str(clean_path),
        "deleted_log": str(deleted_log_path) if drop_invalid else None,
        "half_split": half_file,
        "quarter_split": quarter_file,
        "settings": {
            "min_letters": args.min_letters,
            "wordfreq_threshold": args.wordfreq_threshold,
            "output_delimiter": "<|endoftext|>",
            "strict_english_only": True,
            "no_normalize_ws": args.no_normalize_ws,
        },
        "stats": {
            "raw_count": len(stories),
            "kept_count": len(kept),
            "deleted_count": len(deleted),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
