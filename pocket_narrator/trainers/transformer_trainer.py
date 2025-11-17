"""
Contains the training logic specific to the TransformerModel.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from .base_trainer import AbstractTrainer
from ..models.base_model import AbstractLanguageModel
from ..data_loader import batchify_text

class TransformerTrainer(AbstractTrainer):
    def __init__(self, learning_rate: float = 3e-4, epochs: int = 1, batch_size: int = 32):
        """
        Initializes the Transformer Trainer with its configuration.
        
        Args:
            learning_rate (float): The learning rate for the optimizer.
            epochs (int): The number of times to iterate over the full dataset.
            batch_size (int): The number of sequences to process at once.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._get_device()

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
            x = tokens[:-1]
            y = tokens[1:]
            
            # pad sequences if they are shorter than max_len
            x += [0] * (max_len - len(x)) # assume 0 is the <pad> token ID
            y += [0] * (max_len - len(y))

            batch_x.append(x)
            batch_y.append(y)
            
        return torch.tensor(batch_x, dtype=torch.long).to(self.device), \
               torch.tensor(batch_y, dtype=torch.long).to(self.device)

    def train(self, model: AbstractLanguageModel, tokenizer, train_data: List[str]) -> AbstractLanguageModel:
        """
        The main training loop for the Transformer model.
        """
        print(f"--- Running TransformerTrainer on device: {self.device} ---")
        
        # --- Setup ---
        model.to(self.device) # move the model to the GPU/CPU
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0) # ignore padding token in loss calculation
        
        max_len = model.config['max_len']
        # creating mask for masked self-attention
        causal_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1).to(self.device)

        # --- Training Loop ---
        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")
            
            # create fresh data iterator for each epoch
            train_iterator = batchify_text(train_data, batch_size=self.batch_size, shuffle=True)
            
            pbar = tqdm(train_iterator, desc=f"Epoch {epoch+1} Training")
            for batch_text in pbar:
                # prepare batch
                batch_tokens = tokenizer.encode_batch(batch_text)
                x, y = self._prepare_batch_for_lm(batch_tokens, max_len)
                
                # forward pass
                logits = model(x, mask=causal_mask)
                
                # calculate loss
                # CrossEntropyLoss expects (N, C) and (N), so we reshape
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print("\nTransformer training complete.")
        return model.to("cpu")