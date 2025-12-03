from pathlib import Path
from transformers import DataCollatorForLanguageModeling
from config_utils import load_yaml

from pathlib import Path

from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
from bpe_tokenizer_utils import load_bpe_tokenizer
import inspect  # <--- NEU



class Training:
    def __init__(self, training_config_path: str):
        self.train_cfg = load_yaml(training_config_path)

        model_cfg = load_yaml(self.train_cfg["model_config"])
        tok_cfg = load_yaml(self.train_cfg["tokenizer_config"])

        # 1) LM-Dataset laden
        self.lm_dataset_dir = Path(tok_cfg["lm_dataset_dir"])
        self.lm_dataset = load_from_disk(str(self.lm_dataset_dir))

        # 2) BPE-Tokenizer laden
        self.tokenizer = load_bpe_tokenizer(tok_cfg["save_dir"])

        # 3) GPT-2-Config von scratch mit passender vocab_size
        vocab_size = self.tokenizer.vocab_size
        print(f"Tokenizer vocab_size = {vocab_size}")

        gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=model_cfg.get("n_positions", tok_cfg["block_size"]),
            n_ctx=model_cfg.get("n_ctx", tok_cfg["block_size"]),
            n_embd=model_cfg.get("n_embd", 768),
            n_layer=model_cfg.get("n_layer", 12),
            n_head=model_cfg.get("n_head", 12),
        )

        self.model = GPT2LMHeadModel(gpt2_config)

        # 4) Device wählen
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS backend")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        self.model.to(device)

        # 5) TrainingArguments aus YAML -> aber nur erlaubte Keys übergeben
        ta = self.train_cfg["training_args"]

        # alle möglichen Argumentnamen für deine transformers-Version holen
        sig = inspect.signature(TrainingArguments.__init__)
        allowed_params = set(sig.parameters.keys())

        # Basis-kwargs, die wir gerne hätten:
        base_kwargs = {
            "output_dir": ta["output_dir"],
            "overwrite_output_dir": ta["overwrite_output_dir"],
            "num_train_epochs": ta["num_train_epochs"],
            "per_device_train_batch_size": ta["per_device_train_batch_size"],
            "per_device_eval_batch_size": ta["per_device_eval_batch_size"],
            # moderne Versionen:
            "evaluation_strategy": ta.get("evaluation_strategy", "steps"),
            "eval_steps": ta.get("eval_steps", 100),
            "save_steps": ta["save_steps"],
            "logging_steps": ta["logging_steps"],
            "learning_rate": ta["learning_rate"],
            "warmup_steps": ta["warmup_steps"],
            "save_total_limit": ta["save_total_limit"],
            "fp16": ta["fp16"],
            # Enable wandb logging
            "report_to": ta.get("report_to", "wandb"),
        }

        # auf erlaubte Parameter filtern
        kwargs = {k: v for k, v in base_kwargs.items() if k in allowed_params}

        # Fallback für ältere transformers-Versionen ohne evaluation_strategy:
        if (
            "evaluation_strategy" not in allowed_params
            and "eval_steps" in base_kwargs
            and "do_eval" in allowed_params
        ):
            # aktivier Eval während des Trainings
            kwargs["do_eval"] = True

        self.training_args = TrainingArguments(**kwargs)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def main(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.lm_dataset,
            eval_dataset=self.lm_dataset,
            data_collator=self.data_collator,
        )
        print("Starting training")
        trainer.train()
        print("Training complete, saving model")

        out_dir = Path(self.training_args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out_dir / "model"))
        self.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        print(f"Model + tokenizer saved to {out_dir}")
