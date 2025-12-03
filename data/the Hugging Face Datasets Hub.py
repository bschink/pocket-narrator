from datasets import load_dataset
import json, os

name = "roneneldan/TinyStories"  # or "roneneldan/TinyStoriesInstruct"
ds = load_dataset(name)  # splits: train, validation
os.makedirs("data", exist_ok=True)

def dump(split):
    path = f"data/{name.split('/')[-1]}_{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in ds[split]:
            # the base set has 'text'; instruct has fields like 'Story', 'Words', etc.
            text = row.get("text") or row.get("Story") or ""
            if text.strip():
                f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")
    return path

train_path = dump("train")
val_path   = dump("validation")
print("Wrote:", train_path, val_path)
