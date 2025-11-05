
"""
Optional TinyStories fetcher.

Usage:
  PYTHONPATH=. python3 scripts/fetch_tinystories.py               # will prompt
  PYTHONPATH=. python3 scripts/fetch_tinystories.py --yes         # auto-download
  PYTHONPATH=. python3 scripts/fetch_tinystories.py --dest data/raw/TinyStories
"""

import argparse
import os
import sys

DEFAULT_DEST = "data/raw/TinyStories"

def dir_has_files(path: str) -> bool:
    try:
        return os.path.isdir(path) and any(os.scandir(path))
    except FileNotFoundError:
        return False

def main():
    ap = argparse.ArgumentParser(description="Optionally download TinyStories dataset.")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="Destination directory")
    ap.add_argument("--yes", action="store_true", help="Proceed without prompting")
    args = ap.parse_args()

    dest = args.dest

    # 1) already present?
    if dir_has_files(dest):
        print(f"[info] TinyStories already present at: {dest}. Nothing to do.")
        sys.exit(0)

    # 2) ask user if not --yes
    if not args.yes:
        reply = input(f"[prompt] TinyStories not found at '{dest}'. Download now? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("[info] Skipping download.")
            sys.exit(0)

    # 3) try to download
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        print(f"        details: {e}")
        sys.exit(1)

    os.makedirs(dest, exist_ok=True)
    print(f"[info] Downloading TinyStories to: {dest} ...")
    try:
        snapshot_download(
            repo_id="roneneldan/TinyStories",
            repo_type="dataset",
            local_dir=dest,
            local_dir_use_symlinks=False,   # copy files, safer for sharing/moving
            resume_download=True            # resume if partially present
        )
        print("[success] TinyStories downloaded.")
    except Exception as e:
        print(f"[error] Download failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
