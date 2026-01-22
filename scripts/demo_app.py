import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import argparse
import os

import gradio as gr

from pocket_narrator.models import load_model
from pocket_narrator.tokenizers import get_tokenizer

MODEL_PATH = "models/transformer_28M_higher_lr_full_dataset_too_many_epochs.model"
GENERATION_STRATEGY = "sample"
TOKENIZER_TYPE = "bpe"
TOKENIZER_PATH = "tokenizers/bpe_tokenizer_5k_full_20_rounds"
MAX_NEW_TOKENS = 5000

def tell_story(prompt, temperature):
    try:
        model = load_model(MODEL_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a valid model file. You can create one by running:")
        print("  python scripts/train.py")
        return

    try:
        tokenizer = get_tokenizer(
            tokenizer_type=TOKENIZER_TYPE,
            tokenizer_path=TOKENIZER_PATH
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure a valid tokenizer exists. You can create one by running:")
        print("  python scripts/train.py")
        return
    
    print(f"\nInput prompt: '{prompt}'")
    prompt_tokens_batch = [tokenizer.encode(prompt)]
    
    predict_kwargs = {
        "strategy": GENERATION_STRATEGY,
        "temperature": temperature,
        "max_length": MAX_NEW_TOKENS,
        "top_k": None,
        "top_p": None
    }
    
    predicted_tokens_batch = model.predict_sequence_batch(
        prompt_tokens_batch,
        **predict_kwargs
    )
    predicted_text_batch = tokenizer.decode_batch(predicted_tokens_batch)
    
    generated_continuation = predicted_text_batch[0]
    full_story = prompt + generated_continuation

    if full_story.endswith("<|endoftext|>"):
        full_story = full_story[:-len("<|endoftext|>")]

    return full_story

with gr.Blocks() as demo:
    with gr.Column():
        input_prompt = gr.Textbox(label="Input Prompt", lines=2, elem_classes="big-text")
        temperature = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Temperature")
        generate_btn = gr.Button("Generate")
        
    output_text = gr.Textbox(label="Generated Story", lines=10, elem_classes="big-text")

    generate_btn.click(
        fn=tell_story,
        inputs=[input_prompt, temperature],
        outputs=output_text
    )

demo.launch(css=".big-text textarea { font-size: 1.2rem !important; }")
