"""
The main Transformer model class.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import AbstractLanguageModel
from ..components.base_pos_encoding import AbstractPositionalEncoding
from .transformer_block import TransformerBlock

class TransformerModel(AbstractLanguageModel, nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, 
                 pos_encoding_module: AbstractPositionalEncoding, 
                 transformer_block_template: TransformerBlock):
        AbstractLanguageModel.__init__(self, vocab_size)
        nn.Module.__init__(self)

        d_model = transformer_block_template.attn.d_k * transformer_block_template.attn.n_head

        self.config = {
            "model_type": "transformer",
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_head": transformer_block_template.attn.n_head,
            "max_len": pos_encoding_module.pe.size(1),
            "dropout": transformer_block_template.dropout.p,
            "pos_encoding_type": pos_encoding_module.__class__.__name__,
            "attention_type": transformer_block_template.attn.__class__.__name__,
        }

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = pos_encoding_module
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, transformer_block_template.attn.__class__(
                d_model, 
                transformer_block_template.attn.n_head, 
                transformer_block_template.attn.dropout.p
            ), transformer_block_template.dropout.p)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor = None):
        """The forward pass of the model."""
        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def predict_sequence_batch(self, input_tokens_batch: list[list[int]], **kwargs) -> list[list[int]]:
        """
        Generates a sequence continuation for each prompt in the batch using
        autoregressive decoding.
        """
        was_training = self.training
        self.eval()
        
        max_length = kwargs.get("max_length", 50)
        strategy = kwargs.get("strategy", "greedy")
        eos_token_id = self.config.get("eos_token_id")

        device = next(self.parameters()).device
        
        predictions = []
        for prompt_tokens in input_tokens_batch:
            idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            for _ in range(max_length):
                idx_cond = idx[:, -self.config['max_len']:]
                
                logits = self(idx_cond)
                
                logits = logits[:, -1, :] # (1, vocab_size)
                
                # generation strategy
                if strategy == "sample":
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else: # default to greedy
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
                idx = torch.cat((idx, idx_next), dim=1)
                
                if eos_token_id is not None and idx_next.item() == eos_token_id:
                    break

            generated_list = idx.squeeze(0).tolist()
            predictions.append(generated_list[len(prompt_tokens):])
        
        if was_training:
            self.train()
            
        return predictions

    def save(self, model_path: str):
        """
        Saves the model's configuration and state_dict to a single file.
        """
        print(f"INFO: Saving Transformer model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        save_dict = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(save_dict, model_path)

    @classmethod
    def load(cls, model_path: str, config: dict):
        """
        This method is a placeholder. The master 'load_model' factory now handles
        the full loading process for PyTorch models. This method is only here to
        satisfy the abstract contract, but it should not be called directly.
        """
        raise NotImplementedError("Use the master 'load_model' factory in models/__init__.py to load PyTorch models.")