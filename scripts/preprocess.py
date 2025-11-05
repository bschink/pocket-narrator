

"""
TinyStories CleanUp.

Usage example:
  PYTHONPATH=. python3 scripts/preprocess.py \
  --input data/raw/TinyStories/TinyStories-train.txt \
  --output data/processed/TinyStories/TinyStories-train.bos_eos.txt
"""

import argparse
from pocket_narrator.data_loader import preprocess_txt_to_bos_eos

def main():
    ap = argparse.ArgumentParser(description="Preprocess a .txt corpus into <bos>/<eos>-wrapped documents.")
    ap.add_argument("--input", required=True, help="Path to raw .txt (e.g., TinyStories-train.txt)")
    ap.add_argument("--output", required=True, help="Path to write preprocessed file")
    ap.add_argument("--delimiter", default="<|endoftext|>", help="Document delimiter token in the raw file")
    ap.add_argument("--bos", default="<bos>", help="BOS token to insert")
    ap.add_argument("--eos", default="<eos>", help="EOS token to insert")
    ap.add_argument("--no-normalize-ws", action="store_true", help="Disable whitespace normalization")
    args = ap.parse_args()

    def log(n: int):
        print(f"[preprocess] {n} docs processed...")

    preprocess_txt_to_bos_eos(
        input_path=args.input,
        output_path=args.output,
        bos_token=args.bos,
        eos_token=args.eos,
        delimiter=args.delimiter,
        normalize_ws=not args.no_normalize_ws,
        drop_empty=True,
        progress_log=log,
    )
    print(f"Done. Wrote: {args.output}")

if __name__ == "__main__":
    main()
