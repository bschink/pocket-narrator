# Design goals:Tiny, dependable GPT-style decoder
from dataclasses import dataclass
import math, torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    pdrop: float = 0.0
    tie_weights: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, n_embd, pdrop=0.0, mult=4):
        super().__init__()
        self.fc = nn.Linear(n_embd, mult * n_embd)
        self.proj = nn.Linear(mult * n_embd, n_embd)
        self.drop = nn.Dropout(pdrop)
    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        return self.drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, pdrop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, pdrop)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PocketNarratorModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.pdrop)
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.pdrop) for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        if T > self.cfg.block_size:
            raise ValueError("Sequence length exceeds block_size")
        pos = self.pos_emb[:, :T, :]
        x = self.drop(self.tok_emb(input_ids) + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, eos_token_id=None):
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
        return out
