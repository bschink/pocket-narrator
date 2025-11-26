"""
Contains the training logic specific to the TransformerModel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
from tqdm import tqdm
from typing import List, Optional
from contextlib import nullcontext
import time
from .base_trainer import AbstractTrainer
from ..models.base_model import AbstractLanguageModel
from ..data_loader import batchify_text
from ..evaluate import calculate_perplexity

class TransformerTrainer(AbstractTrainer):
    def __init__(self, 
                 learning_rate: float = 3e-4, 
                 epochs: int = 1, 
                 batch_size: int = 128,
                 weight_decay: float = 0.1,
                 grad_clip: float = 1.0,
                 warmup_steps: int = 0,
                 use_amp: bool = False,
                 kv_caching_enabled: bool = False):
        """
        Initializes the Transformer Trainer with its configuration.
        
        Args:
            learning_rate (float): The learning rate for the optimizer.
            epochs (int): The number of times to iterate over the full dataset.
            batch_size (int): The number of sequences to process at once.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            grad_clip (float): Maximum gradient norm for gradient clipping.
            warmup_steps (int): Number of warmup steps for learning rate scheduling.
            use_amp (bool): Whether to use automatic mixed precision for training.
            kv_caching_enabled (bool): Whether KV caching is enabled during generation.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.kv_caching_enabled = kv_caching_enabled
        self.device = self._get_device()
        # AMP is only supported on CUDA
        self.use_amp = use_amp and self.device == "cuda"

        # padding index used in batches
        self.pad_token_id = 0

        # Only create a GradScaler when we are actually using AMP on CUDA.
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None
        

    def _get_device(self) -> str:
        """Automatically select a device to run on."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _prepare_batch_for_lm(self, batch_tokens: List[List[int]], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a batch for language modeling.
        - Truncates/pads sequences to a fixed max_len.
        - Creates input (x) and target (y) tensors, where y is x shifted by one.
        - Moves tensors to the correct device.
        """
        batch_x, batch_y = [], []
        for tokens in batch_tokens:
            # truncate to max_len + 1
            tokens = tokens[:max_len + 1]
            if len(tokens) < 2: continue

            x = tokens[:-1]
            y = tokens[1:]
            
            # pad sequences if they are shorter than max_len
            pad_len = max_len - len(x)
            x += [0] * pad_len # assume 0 is the <pad> token ID
            y += [0] * pad_len

            batch_x.append(x)
            batch_y.append(y)

        if not batch_x:
            return None, None
            
        return torch.tensor(batch_x, dtype=torch.long).to(self.device), \
               torch.tensor(batch_y, dtype=torch.long).to(self.device)
    
    def _get_cosine_schedule_with_warmup(self, optimizer, num_training_steps):
        """
        Creates a learning rate scheduler with linear warmup and cosine decay.
        """
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, num_training_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @torch.no_grad()
    def calculate_validation_loss(self, model: nn.Module, tokenizer, val_data: List[str]) -> float:
        """
        Computes rigorous validation loss (NLL) for Perplexity.
        """
        was_training = model.training
        model.to(self.device)
        model.eval()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        max_len = model.config['max_len']
        causal_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1).to(self.device)
        
        total_nll = 0.0
        total_tokens = 0
        
        val_iterator = batchify_text(val_data, batch_size=self.batch_size, shuffle=False)
        
        for batch_text in val_iterator:
            batch_tokens = tokenizer.encode_batch(batch_text)
            x, y = self._prepare_batch_for_lm(batch_tokens, max_len)
            
            if x is None: continue

            # MPS doesn't support float16 autocast, so use no-op context for MPS
            if self.use_amp and self.device == "cuda":
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                logits, _ = model(x, mask=causal_mask, use_cache=False)
                loss_sum = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            num_valid_tokens = (y.view(-1) != 0).sum().item()
            total_nll += loss_sum.item()
            total_tokens += num_valid_tokens
            
        if was_training: model.train()
        
        if total_tokens == 0: return float('inf')
        return total_nll / total_tokens
    
    def compute_batch_loss(self, model, tokenizer, train_data: List[str]) -> torch.Tensor:
        """
        Sample one random batch from train_data and compute cross-entropy loss.
        """
        max_len = model.config["max_len"]

        # get a random batch of text
        batch_iterator = batchify_text(train_data, batch_size=self.batch_size, shuffle=True)
        batch_text = next(batch_iterator)

        # tokenize and prepare for LM
        batch_tokens = tokenizer.encode_batch(batch_text)
        input_batch, target_batch = self._prepare_batch_for_lm(batch_tokens, max_len)

        if input_batch is None:
            # degenerate case: all sequences too short
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # choose AMP or no AMP
        if self.use_amp and self.device == "cuda":
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        # forward pass
        with amp_ctx:
            out = model(input_batch)

            # model might return (logits, cache) or just logits
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_batch.view(-1),
                ignore_index=self.pad_token_id,
            )

        return loss


    def train(self, model: AbstractLanguageModel, tokenizer, train_data: List[str], val_data: List[str] = None, batch_size: int = None) -> AbstractLanguageModel:
        """
        The main training loop for the Transformer model.
        optional validation data can be provided to report perplexity during training.
        If batch_size is provided, it overrides the trainer's default batch_size.
        """
        # Use provided batch_size if specified, otherwise use instance batch_size
        if batch_size is not None:
            self.batch_size = batch_size
            
        print(f"--- Running TransformerTrainer on device: {self.device} ---")
        print(f"INFO: Using batch_size={self.batch_size}")
        
        # --- Setup ---
        model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # calculate approx total steps for scheduler
        approx_steps_per_epoch = len(train_data) // self.batch_size
        steps_per_epoch = max(1, approx_steps_per_epoch)
        total_steps = steps_per_epoch * self.epochs
        scheduler = self._get_cosine_schedule_with_warmup(optimizer, total_steps)

        step_counter = 0
        best_val_loss = float('inf')
        training_start_time = time.time()

        # --- Training Loop ---
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{self.epochs}")
            for step_idx in pbar:
                optimizer.zero_grad()

                loss = self.compute_batch_loss(model, tokenizer, train_data)
                loss.backward()

                # Compute gradient norm before clipping
                grad_norm = None
                if self.grad_clip:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # --- wandb dynamic log with LR and gradient norm ---
                log_dict = {
                    "train/loss": loss.item(),
                    "train/perplexity": math.exp(loss.item()),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                
                # Log gradient norm if available
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                
                wandb.log(log_dict)
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / steps_per_epoch

            # Compute validation loss if val_data is provided
            if val_data is not None:
                val_loss = self.calculate_validation_loss(model, tokenizer, val_data)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                val_perplexity = math.exp(val_loss) if val_loss != float('inf') else float('inf')
            else:
                val_loss = None
                val_perplexity = None

            # Extra epoch-level logs
            epoch_log = {
                "epoch/loss_avg": avg_loss,
                "epoch/perplexity_avg": math.exp(avg_loss),
                "epoch/number": epoch + 1,
            }
            if val_loss is not None:
                epoch_log["epoch/val_loss"] = val_loss
                epoch_log["epoch/val_perplexity"] = val_perplexity
            
            wandb.log(epoch_log)

        # --- Final timing and summary stats ---
        training_duration = time.time() - training_start_time
        
        print("\nTransformer training complete.")
        
        # Log final stats to wandb.summary for easy comparison
        wandb.summary["training/best_val_loss"] = best_val_loss if best_val_loss != float('inf') else None
        wandb.summary["training/final_val_perplexity"] = math.exp(best_val_loss) if best_val_loss != float('inf') else None
        wandb.summary["training/total_duration_seconds"] = training_duration
        wandb.summary["training/num_model_params"] = sum(p.numel() for p in model.parameters())
        wandb.summary["training/device"] = self.device
        wandb.summary["training/vocab_size"] = tokenizer.get_vocab_size()
        
        return model.to("cpu")