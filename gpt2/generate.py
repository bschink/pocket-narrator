from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch
from pathlib import Path


def load(model_dir, tokenizer_dir, device: torch.device):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    device: torch.device = torch.device("cpu"),
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device) if "attention_mask" in enc else None

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_new_tokens,
            do_sample=True,
            temperature=0.8,          
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,   # prevents 3-gram repetitions
            repetition_penalty=1.2,   # Repeat punished for repetition
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Wenn MPS zickt: einfach device = torch.device("cpu") erzwingen
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "results" / "gpt2_medium" / "model"
    tokenizer_dir = base_dir / "results" / "gpt2_medium" / "tokenizer"

    model, tokenizer = load(str(model_dir), str(tokenizer_dir), device)

    prompt = "Once upon a time "
    text = generate(model, tokenizer, prompt, max_new_tokens=80, device=device)

    print("\n--- PROMPT ---")
    print(prompt)
    print("\n--- GENERATED ---")
    print(text)
