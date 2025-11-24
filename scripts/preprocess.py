"""
Strict-English Dataset Preprocessing Script.

- Splits raw .txt into stories using default <|endoftext|>.
- Removes a story if ANY of the following holds:
    * has < min_letters ASCII letters
    * contains ANY non-ASCII characters (Chinese, umlauts, accents...)
    * contains ANY non-English word (based on wordfreq zipf thresholdfloat that we specify)
    * is a duplicate (exact text match after normalization)
- Formats output as:
        STORY_TEXT <|endoftext|>
- Automatically creates:
        <input>.clean.txt
        <input>.clean.half.txt
        <input>.clean.quarted.txt
        <input>.clean.deleted.txt
        <input>.clean.manifest.json

- Also generates JSON manifest and CLI summary summarizing:
    * total stories/characters kept & removed
    * per-reason stats (invalid_chars, non_english, duplicate)
      with story/char counts and percentages
    * frequency table of all non-English words encountered
    * frequency table of all invalid characters encountered


Usage example:
PYTHONPATH=. python3 scripts/preprocess.py \
--input data/raw/TinyStories/TinyStoriesV2-GPT4-train.txt \
--output data/processed/TinyStories/
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import re
import random
from collections import Counter

# tqdm import with fallback
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# wordfreq for English detection
try:
    from wordfreq import zipf_frequency
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "wordfreq is required for non-English detection. "
        "Install it with: pip install wordfreq"
    ) from e


ALLOWED_CHARS_REGEX = re.compile(
    r"[A-Za-z0-9\s.,;:'\"?!()\-\[\]/’‘“”…—`$+]"
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
) -> tuple[list[str], list[str]]:
    """
    Strict-English invalid story detection:

    -> Reject ANY non-ASCII characters (Chinese, Cyrillic, äöü, é, etc.)
    -> Reject ANY non-allowed ASCII punctuation (using ALLOWED_CHARS_REGEX)
    -> Require minimum number of ASCII alphabetic characters

    Returns:
      (reasons, invalid_chars_list)
    """
    reasons: list[str] = []
    invalid_chars: list[str] = []

    # Count ASCII alphabetic letters only
    ascii_letter_count = sum(ch.isascii() and ch.isalpha() for ch in text)
    if ascii_letter_count < min_letters:
        reasons.append(f"too_few_letters({ascii_letter_count})")

    # Reject ANY non-ASCII character or disallowed ASCII char
    for ch in text:
        if ch.isspace():
            continue

        if not ch.isascii():
            reasons.append(f"invalid_chars(non_ascii:{repr(ch)})")
            invalid_chars.append(ch)
            return reasons, invalid_chars

        if not ALLOWED_CHARS_REGEX.fullmatch(ch):
            reasons.append(f"invalid_chars({repr(ch)})")
            invalid_chars.append(ch)
            return reasons, invalid_chars

    return reasons, invalid_chars


def is_english_word(word: str, threshold: float) -> bool:
    """Use wordfreq zipf frequency to test English-ness."""
    if not word:
        return False
    return zipf_frequency(word.lower(), "en") >= threshold


def detect_non_english_strict(
    text: str,
    wordfreq_threshold: float,
) -> tuple[list[str], list[str]]:
    """
    STRICT rule:

    If ANY token is not English → story removed.

    Steps:
      - Extract ONLY ASCII alphabetic tokens
      - Use wordfreq to classify English words
      - Reject if ANY fail

    Returns:
      (reasons, bad_words_list)
    """
    tokens = re.findall(r"[A-Za-z]+", text)

    if not tokens:
        return ["non_english_words(no_ascii_tokens)"], []

    bad = [t for t in tokens if not is_english_word(t, wordfreq_threshold)]

    if bad:
        sample = ", ".join(bad[:5])
        return [f"non_english_words({sample})"], bad

    return [], []


def build_duplicate_map(stories: list[str]) -> dict[int, list[str]]:
    """Detect exact-text duplicates after whitespace normalization & lowercasing."""
    seen: dict[str, int] = {}
    dup_reasons: dict[int, list[str]] = {}
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

    ap.add_argument(
        "--input",
        required=True,
        help="Path to raw .txt file",
    )

    ap.add_argument(
        "--output",
        required=True,
        help="Destination folder or file",
    )

    # input delimiter
    ap.add_argument(
        "--delimiter",
        default="<|endoftext|>",
        help="Raw-story delimiter (default: <|endoftext|>)",
    )

    # whitespace
    ap.add_argument(
        "--no-normalize-ws",
        action="store_true",
        help="Disable whitespace normalization",
    )

    # minimum letters
    ap.add_argument(
        "--min-letters",
        type=int,
        default=100,
        help="Minimum ASCII letters per story (default: 100)",
    )

    # strict English settings
    ap.add_argument(
        "--wordfreq-threshold",
        type=float,
        default=2.5,
        help="Minimum zipf freq for English words (0-7 where 7 only allows the most common words like 'the' 'and')",
    )

    # keep or drop invalid
    ap.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep invalid stories (do not delete). Default: drop invalid stories.",
    )

    # splitting
    ap.add_argument(
        "--no-splits",
        action="store_true",
        help="Disable half/quarter splits",
    )

    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )

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

    # compute original char count
    original_story_count = len(stories)
    original_char_count = sum(len(s) for s in stories)

    if not args.no_normalize_ws:
        print("[step 1] Normalizing whitespace…")
        stories = [normalize_whitespace(s) for s in tqdm(stories, desc="normalize_ws")]

    # ------------------------------------------
    # Step 2 — strict invalid checks + non-English + duplicates
    # ------------------------------------------

    invalid_reasons: dict[int, list[str]] = {}
    non_english_reasons: dict[int, list[str]] = {}
    dup_reasons: dict[int, list[str]] = {}

    # Counters for invalid chars / non-English words
    invalid_char_counts: Counter[str] = Counter()
    non_english_word_counts: Counter[str] = Counter()

    print("[step 2a] Checking strict invalid chars + length")
    for idx, s in enumerate(tqdm(stories, desc="invalid_check")):
        reasons, bad_chars = detect_invalid_story(s, args.min_letters)
        if reasons:
            invalid_reasons[idx] = reasons
        if bad_chars:
            invalid_char_counts.update(bad_chars)

    print("[step 2b] Checking duplicates")
    dup_reasons = build_duplicate_map(stories)

    print("[step 2c] Checking strict non-English (ANY bad word removes story)")
    for idx, s in enumerate(tqdm(stories, desc="non_english_check")):
        reasons, bad_words = detect_non_english_strict(s, args.wordfreq_threshold)
        if reasons:
            non_english_reasons[idx] = reasons
        if bad_words:
            non_english_word_counts.update(w.lower() for w in bad_words)

    # merge reasons
    all_reasons: dict[int, list[str]] = {}
    for idx in range(len(stories)):
        temp: list[str] = []
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

    kept: list[str] = []
    deleted: list[tuple[int, str, list[str]]] = []

    for idx, s in enumerate(stories):
        r = all_reasons.get(idx, [])
        if r and drop_invalid:
            deleted.append((idx, s, r))
        else:
            kept.append(s)

    kept_story_count = len(kept)
    kept_char_count = sum(len(s) for s in kept)
    deleted_story_count = len(deleted)
    deleted_char_count = original_char_count - kept_char_count

    print(f"[step 3] Kept {kept_story_count} stories. Deleted {deleted_story_count}.")

    # write deleted log
    if drop_invalid and deleted:
        deleted_log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[step 3] Writing deleted log → {deleted_log_path}")
        with open(deleted_log_path, "w", encoding="utf-8") as f:
            for idx, text, reasons in deleted:
                f.write(f"### STORY {idx} | reasons: {', '.join(reasons)}\n{text}\n\n")

    # ------------------------------------------
    # Step 3b — Per-reason stats (stories + chars)
    # ------------------------------------------

    reason_categories = ["invalid_chars", "non_english", "duplicate"]

    category_story_counts = {cat: 0 for cat in reason_categories}
    category_char_counts = {cat: 0 for cat in reason_categories}

    for idx, text, reasons in deleted:
        L = len(text)
        has_invalid = any(r.startswith("invalid_chars") for r in reasons)
        has_noneng = any(r.startswith("non_english") for r in reasons)
        has_dup = any(r.startswith("duplicate_of_") for r in reasons)

        if has_invalid:
            category_story_counts["invalid_chars"] += 1
            category_char_counts["invalid_chars"] += L
        if has_noneng:
            category_story_counts["non_english"] += 1
            category_char_counts["non_english"] += L
        if has_dup:
            category_story_counts["duplicate"] += 1
            category_char_counts["duplicate"] += L

    # compute percentages
    def pct(part: int, whole: int) -> float:
        if whole == 0:
            return 0.0
        return (part / whole) * 100.0

    total_stories_pct_removed = pct(deleted_story_count, original_story_count)
    total_chars_pct_removed = pct(deleted_char_count, original_char_count)

    category_stats = {}
    for cat in reason_categories:
        s_count = category_story_counts[cat]
        c_count = category_char_counts[cat]
        category_stats[cat] = {
            "story_count": s_count,
            "story_pct_of_all": pct(s_count, original_story_count),
            "char_count": c_count,
            "char_pct_of_all": pct(c_count, original_char_count),
        }

    # ------------------------------------------
    # Step 4 — Output cleaned
    # ------------------------------------------

    print(f"[step 4] Writing cleaned output → {clean_path}")
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    output_delim = "<|endoftext|>"
    with open(clean_path, "w", encoding="utf-8") as f:
        for s in tqdm(kept, desc="write_clean"):
            f.write(format_story_for_output(s, output_delim) + "\n")

    # ------------------------------------------
    # Step 5 — Splits
    # ------------------------------------------

    half_file = None
    quarter_file = None

    if make_splits and kept:
        print("[step 5] Creating 1/2 and 1/4 splits…")
        rng = random.Random(args.split_seed)
        indices = list(range(len(kept)))
        rng.shuffle(indices)

        n = len(kept)
        half_n = n // 2
        quarter_n = n // 4

        half_file = str(half_path)
        quarter_file = str(quarter_path)

        half_path.parent.mkdir(parents=True, exist_ok=True)
        quarter_path.parent.mkdir(parents=True, exist_ok=True)

        with open(half_file, "w", encoding="utf-8") as f_half:
            for i in indices[:half_n]:
                f_half.write(format_story_for_output(kept[i], output_delim) + "\n")

        with open(quarter_file, "w", encoding="utf-8") as f_quarter:
            for i in indices[:quarter_n]:
                f_quarter.write(format_story_for_output(kept[i], output_delim) + "\n")

        print(f"[step 5] half → {half_file}")
        print(f"[step 5] quarter → {quarter_file}")
    else:
        if not kept:
            print("[step 5] No kept stories; splits skipped.")
        else:
            print("[step 5] Splits disabled via --no-splits.")

    # ------------------------------------------
    # Step 6 — Manifest
    # ------------------------------------------

    print(f"[step 6] Writing manifest → {manifest_path}")

    manifest = {
        "input": args.input,
        "cleaned": str(clean_path),
        "deleted_log": str(deleted_log_path) if drop_invalid and deleted else None,
        "half_split": half_file,
        "quarter_split": quarter_file,
        "settings": {
            "delimiter": args.delimiter,
            "min_letters": args.min_letters,
            "wordfreq_threshold": args.wordfreq_threshold,
            "output_delimiter": "<|endoftext|>",
            "strict_english_only": True,
            "normalize_whitespace": not args.no_normalize_ws,
            "drop_invalid": drop_invalid,
        },
        "stats": {
            "raw_story_count": original_story_count,
            "raw_char_count": original_char_count,
            "kept_story_count": kept_story_count,
            "kept_char_count": kept_char_count,
            "deleted_story_count": deleted_story_count,
            "deleted_char_count": deleted_char_count,
            "deleted_story_pct": total_stories_pct_removed,
            "deleted_char_pct": total_chars_pct_removed,
            "per_reason": category_stats,
            "non_english_word_counts": dict(non_english_word_counts),
            "invalid_char_counts": dict(invalid_char_counts),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ------------------------------------------
    # Step 7 — CLI summary
    # ------------------------------------------

    print("\n[summary]")
    print(f"  Original stories : {original_story_count}")
    print(f"  Kept stories     : {kept_story_count}")
    print(f"  Deleted stories  : {deleted_story_count} ({total_stories_pct_removed:.2f}%)")
    print(f"  Original chars   : {original_char_count}")
    print(f"  Kept chars       : {kept_char_count}")
    print(f"  Deleted chars    : {deleted_char_count} ({total_chars_pct_removed:.2f}%)")

    print("\n  Deletion breakdown by reason (over all original stories/characters):")
    for cat in reason_categories:
        cs = category_stats[cat]
        print(
            f"    {cat}: "
            f"{cs['story_count']} stories ({cs['story_pct_of_all']:.2f}%), "
            f"{cs['char_count']} chars ({cs['char_pct_of_all']:.2f}%)"
        )

    # Invalid chars & non-English words summary
    print("\n  Invalid characters encountered:")
    if invalid_char_counts:
        for ch, cnt in sorted(invalid_char_counts.items(), key=lambda x: -x[1]):
            # show repr to visualize spaces/odd chars
            print(f"    {repr(ch)} : {cnt}")
    else:
        print("    (none)")


    print("\nDone.")


if __name__ == "__main__":
    main()
