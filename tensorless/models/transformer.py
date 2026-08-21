"""A small, dependency-free (beyond PyTorch) GPT-style decoder transformer.

Used for:
  - "text-generation": next-token prediction over the char vocabulary
  - "text-classification": same backbone, with a classification head on
    the final token's hidden state instead of a language-modeling head

Kept intentionally compact -- this is not meant to compete with
production LLM training frameworks, it's meant to give Tensorless a real,
working, from-scratch model that trains fast enough on CPU for the
"zero setup" experience to actually be pleasant.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=attn_mask is None
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * ff_mult)
        self.fc2 = nn.Linear(d_model * ff_mult, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """Decoder-only transformer usable for LM or sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        max_seq_len: int,
        task: str = "text-generation",
        n_classes: int = 0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.task = task
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, ff_mult, dropout) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        if task == "text-generation":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # weight tying
        elif task == "text-classification":
            assert n_classes > 0, "n_classes must be set for text-classification"
            self.head = nn.Linear(d_model, n_classes)
        else:
            raise ValueError(f"Unsupported task for TinyTransformer: {task}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        attn_mask = None
        if attention_mask is not None:
            # Combine causal mask with padding mask.
            causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
            pad = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # B,1,1,T
            attn_mask = (causal.unsqueeze(0).unsqueeze(0) & pad)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)

        if self.task == "text-generation":
            return self.head(x)  # B, T, vocab_size
        else:
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            else:
                lengths = torch.full((B,), T - 1, device=input_ids.device)
            pooled = x[torch.arange(B, device=input_ids.device), lengths]
            return self.head(pooled)  # B, n_classes

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            cond = input_ids[:, -self.max_seq_len:]
            logits = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return input_ids
