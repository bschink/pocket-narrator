"""
Contains the training logic specific to the TransformerModel.
"""
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from typing import List, Optional
from .base_trainer import AbstractTrainer
from ..models.base_model import AbstractLanguageModel
from ..data_loader import batchify_text
from ..evaluate import calculate_perplexity

class TransformerTrainer(AbstractTrainer):
    def __init__(self, 
                 learning_rate: float = 3e-4, 
                 epochs: int = 1, 
                 batch_size: int = 32,
                 weight_decay: float = 0.1,
                 grad_clip: float = 1.0,
                 warmup_steps: int = 100,
                 use_amp: bool = True):
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
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.device = self._get_device()
        # AMP is only supported on CUDA
        self.use_amp = use_amp and self.device == "cuda"
        # GradScaler requires "cuda" or "cpu" device_type (not "mps")
        scaler_device = "cuda" if self.use_amp else "cpu"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.use_amp)

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

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits, _ = model(x, mask=causal_mask, use_cache=False)
                loss_sum = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            num_valid_tokens = (y.view(-1) != 0).sum().item()
            total_nll += loss_sum.item()
            total_tokens += num_valid_tokens
            
        if was_training: model.train()
        
        if total_tokens == 0: return float('inf')
        return total_nll / total_tokens

    def train(self, model: AbstractLanguageModel, tokenizer, train_data: List[str], val_data: List[str] = None) -> AbstractLanguageModel:
        """
        The main training loop for the Transformer model.
        optional validation data can be provided to report perplexity during training.
        """
        print(f"--- Running TransformerTrainer on device: {self.device} ---")
        
        # --- Setup ---
        model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # calculate approx total steps for scheduler
        approx_steps_per_epoch = len(train_data) // self.batch_size
        total_steps = approx_steps_per_epoch * self.epochs
        scheduler = self._get_cosine_schedule_with_warmup(optimizer, total_steps)

        loss_fn = nn.CrossEntropyLoss(ignore_index=0) # ignore padding token in loss calculation
        max_len = model.config['max_len']
        causal_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1).to(self.device)

        step_counter = 0

        # --- Training Loop ---
        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")
            
            # create fresh data iterator for each epoch
            train_iterator = batchify_text(train_data, batch_size=self.batch_size, shuffle=True)
            
            pbar = tqdm(train_iterator, total=approx_steps_per_epoch, desc=f"Epoch {epoch+1} Training")
            for batch_text in pbar:
                # prepare batch
                batch_tokens = tokenizer.encode_batch(batch_text)
                x, y = self._prepare_batch_for_lm(batch_tokens, max_len)

                if x is None: continue

                optimizer.zero_grad(set_to_none=True)
                
                # forward pass
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    logits, _ = model(x, mask=causal_mask, use_cache=False)
                    # CrossEntropyLoss expects (N, C) and (N), so we reshape
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # backward pass
                if self.use_amp:
                    # AMP-enabled path (CUDA only)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard path (CPU/MPS or no AMP)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                
                scheduler.step()

                step_counter += 1
                
                # Log Metrics
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            # end of epoch validation
            if val_data:
                val_loss = self.calculate_validation_loss(model, tokenizer, val_data)
                ppl = calculate_perplexity(val_loss)
                print(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
        
        print("\nTransformer training complete.")
        return model.to("cpu")