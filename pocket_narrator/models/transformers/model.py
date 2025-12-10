"""
The main Transformer model class.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from ..base_model import AbstractLanguageModel
from ..components.positional_encoding import SinusoidalPositionalEncoding, RotaryPositionalEncoding
from .attention import MultiHeadSelfAttention, LinearAttention
from .transformer_block import TransformerBlock

class TransformerModel(AbstractLanguageModel, nn.Module):
    def __init__(self, vocab_size: int, blocks: nn.ModuleList, config: dict,
                 pos_encoding_module: AbstractLanguageModel = None):
        AbstractLanguageModel.__init__(self, vocab_size)
        nn.Module.__init__(self)

        self.config = config
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
        d_model = kwargs.get('d_model', 128)
        n_layers = kwargs.get('n_layers', 2)
        n_head = kwargs.get('n_head', 2)
        max_len = kwargs.get('max_len', 128)
        dropout = kwargs.get('dropout', 0.1)
        pos_encoding_type = kwargs.get("pos_encoding_type", "sinusoidal")
        attention_type = kwargs.get("attention_type", "multi_head")
        activation_type = kwargs.get("activation_type", "gelu")

        # positional encodings
        additive_pos_encoding = None
        rotary_pos_encoding = None
        if pos_encoding_type == "sinusoidal":
            additive_pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding_type == "rope":
            assert (d_model // n_head) % 2 == 0, "head dim must be even for RoPE"
            rotary_pos_encoding = RotaryPositionalEncoding(d_model // n_head, max_len)
        else:
            raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")

        # transformer-specific components
        blocks = nn.ModuleList()
        for _ in range(n_layers):
            if attention_type == "multi_head":
                attention_module = MultiHeadSelfAttention(d_model, n_head, dropout, pos_encoding_module=rotary_pos_encoding)
            elif attention_type == "linear":
                # disable RoPE module passing for Linear Attention
                attention_module = LinearAttention(d_model, n_head, dropout, pos_encoding_module=None)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")
            
            blocks.append(TransformerBlock(d_model, attention_module, dropout, activation_type=activation_type))

        # store configuration
        config = {
            "model_type": "transformer", "vocab_size": vocab_size, "d_model": d_model,
            "n_layers": n_layers, "n_head": n_head, "max_len": max_len, "dropout": dropout,
            "pos_encoding_type": pos_encoding_type, "attention_type": attention_type, 
            "activation_type": activation_type, "eos_token_id": kwargs.get("eos_token_id")
        }
        
        return cls(vocab_size, blocks, config, pos_encoding_module=additive_pos_encoding)

    def forward(self, 
                idx: torch.Tensor, 
                mask: torch.Tensor = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                is_causal: bool = True):
        """
        Args:
            idx: Input tokens. Shape (B, L).
            mask: Attention mask.
            past_key_values: List of (K, V) tuples from previous step.
            use_cache: If True, returns new key_values. If False, returns None for KV.
            is_causal: If True, applies causal masking. Should be True for prefill/non-cached,
                      False for single-token decode with cache.
        """
        x = self.token_embedding(idx)
        if self.pos_encoding is not None:
            offset = past_key_values[0][0].size(2) if past_key_values is not None else 0
            x = self.pos_encoding(x, offset=offset)

        present_key_values = [] if use_cache else None
    
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            x, layer_present = block(x, mask=mask, layer_past=layer_past, is_causal=is_causal)
            
            if use_cache:
                present_key_values.append(layer_present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, present_key_values

    @torch.no_grad()
    def predict_sequence_batch(self, input_tokens_batch: list[list[int]], **kwargs) -> list[list[int]]:
        """
        Generates a sequence continuation for each prompt in the batch.
        """
        was_training = self.training
        self.eval()
        
        max_new_tokens = kwargs.get("max_length", 50)
        strategy = kwargs.get("strategy", "greedy")
        use_cache = kwargs.get("use_cache", False)
        temperature = kwargs.get("temperature", 1.0)
        eos_token_id = self.config.get("eos_token_id")
        device = next(self.parameters()).device
        max_context_len = self.config['max_len']
        
        results = []

        for prompt_tokens in input_tokens_batch:
            if not prompt_tokens:
                results.append([])
                continue
            
            generated = list(prompt_tokens)
            
            # with kv caching
            if use_cache:
                ctx_tokens = generated[-max_context_len:]
                idx = torch.tensor([ctx_tokens], dtype=torch.long, device=device)
                
                logits, past_key_values = self.forward(idx, use_cache=True, is_causal=True)
                next_token_logits = logits[:, -1, :]
                
                for _ in range(max_new_tokens):
                    idx_next = self._sample_token(next_token_logits, strategy, temperature)
                    
                    token_int = idx_next.item()
                    generated.append(token_int)
                    if eos_token_id is not None and token_int == eos_token_id: break
                    if len(generated) >= max_context_len: break

                    logits, past_key_values = self.forward(idx_next, past_key_values=past_key_values, use_cache=True, is_causal=False)
                    next_token_logits = logits[:, -1, :]

            # without kv caching
            else:
                for _ in range(max_new_tokens):
                    ctx_tokens = generated[-max_context_len:]
                    idx = torch.tensor([ctx_tokens], dtype=torch.long, device=device)
                    
                    logits, _ = self.forward(idx, use_cache=False, is_causal=True)
                    next_token_logits = logits[:, -1, :]
                    
                    idx_next = self._sample_token(next_token_logits, strategy, temperature)
                    
                    token_int = idx_next.item()
                    generated.append(token_int)
                    if eos_token_id is not None and token_int == eos_token_id: break

            results.append(generated[len(prompt_tokens):])

        if was_training:
            self.train()
            
        return results

    def _sample_token(self, logits, strategy, temperature=1.0):
        """Helper for sampling strategy with temperature scaling"""
        
        # Apply temperature scaling (only affects sampling, not greedy)
        if strategy == "sample" and temperature != 1.0:
            logits = logits / temperature
        
        # Sample based on strategy
        if strategy == "sample":
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

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