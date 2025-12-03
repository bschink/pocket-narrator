# prepare_eval_data.py
from pathlib import Path
from datasets import load_dataset


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # TinyStories laden
    print("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", "default", split="train")

    # wie viele Beispiele für Eval?
    num_examples = 100
    ds = ds.select(range(num_examples))

    prompts_path = data_dir / "prompts.txt"
    refs_path = data_dir / "references.txt"

    with prompts_path.open("w", encoding="utf-8") as f_p, refs_path.open(
        "w", encoding="utf-8"
    ) as f_r:
        for i, ex in enumerate(ds):
            full = ex["text"].strip()

            # ganz simple Heuristik: erste "Satzhälfte" als Prompt
            # (bis zum ersten Punkt)
            parts = full.split(".")
            first = parts[0].strip()
            if not first:
                first = full[:80]
            if not first.endswith("."):
                first += "."

            prompt = first  # du kannst auch "Continue the story: ..." davor setzen
            reference = full.replace("\n", " ").strip()

            f_p.write(prompt + "\n")
            f_r.write(reference + "\n")

    print(f"Wrote {num_examples} prompts to {prompts_path}")
    print(f"Wrote {num_examples} references to {refs_path}")


if __name__ == "__main__":
    main()
