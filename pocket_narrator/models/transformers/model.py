"""
The main Transformer model class.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import AbstractLanguageModel
from ..components.positional_encoding import SinusoidalPositionalEncoding, RotaryPositionalEncoding
from .attention import MultiHeadSelfAttention
from .transformer_block import TransformerBlock

class TransformerModel(AbstractLanguageModel, nn.Module):
    def __init__(self, vocab_size: int, blocks: nn.ModuleList, config: dict,
                 pos_encoding_module: AbstractLanguageModel = None):
        AbstractLanguageModel.__init__(self, vocab_size)
        nn.Module.__init__(self)

        self.config = config

        # architecture is assembled
        self.token_embedding = nn.Embedding(vocab_size, config['d_model'])
        self.pos_encoding = pos_encoding_module
        self.blocks = blocks
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], vocab_size, bias=False)

    @classmethod
    def from_config(cls, vocab_size: int, **kwargs):
        """
        The factory for creating a TransformerModel from a configuration dictionary.
        This method handles the entire assembly process.
        """
        # hyperparameters
        d_model = kwargs.get('d_model', 256)
        n_layers = kwargs.get('n_layers', 4)
        n_head = kwargs.get('n_head', 4)
        max_len = kwargs.get('max_len', 256)
        dropout = kwargs.get('dropout', 0.1)
        pos_encoding_type = kwargs.get("pos_encoding_type", "sinusoidal")
        attention_type = kwargs.get("attention_type", "multi_head")

        # positional encodings
        additive_pos_encoding = None
        rotary_pos_encoding = None
        if pos_encoding_type == "sinusoidal":
            additive_pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding_type == "rope":
            assert (d_model // n_head) % 2 == 0, "For RoPE, head dimension (d_model/n_head) must be even."
            rotary_pos_encoding = RotaryPositionalEncoding(d_model // n_head, max_len)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        # transformer-specific components
        blocks = nn.ModuleList()
        for _ in range(n_layers):
            if attention_type == "multi_head":
                attention_module = MultiHeadSelfAttention(d_model, n_head, dropout, pos_encoding_module=rotary_pos_encoding)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")
            
            blocks.append(TransformerBlock(d_model, attention_module, dropout))

        # store configuration
        config = {
            "model_type": "transformer", "vocab_size": vocab_size, "d_model": d_model,
            "n_layers": n_layers, "n_head": n_head, "max_len": max_len, "dropout": dropout,
            "pos_encoding_type": pos_encoding_type, "attention_type": attention_type,
        }
        
        return cls(vocab_size, blocks, config, pos_encoding_module=additive_pos_encoding)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor = None):
        """The forward pass of the model."""
        x = self.token_embedding(idx)
        if self.pos_encoding is not None:
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
        the full loading process. This method is only here to
        satisfy the abstract contract, but it should not be called directly.
        """
        raise NotImplementedError("Use the master 'load_model' factory.")