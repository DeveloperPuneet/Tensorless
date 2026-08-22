"""Compact NumPy text model retaining Tensorless transformer behavior."""

from __future__ import annotations

from typing import Optional
import numpy as np

from ..engine import Module, Parameter, softmax_cross_entropy


def _dropout(value, probability, training):
    if not training or probability <= 0:
        return value, None
    keep = np.random.random(value.shape) >= probability
    return value * keep / (1.0 - probability), keep


class TransformerBlock(Module):
    def __init__(self, d_model, heads, ff_mult, dropout):
        if d_model % heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads, self.head_dim, self.dropout = heads, d_model // heads, dropout
        rng = np.random.default_rng()
        self.qkv_weight = Parameter(rng.normal(0, .02, (d_model, 3 * d_model)).astype(np.float32))
        self.qkv_bias = Parameter(np.zeros(3 * d_model, dtype=np.float32))
        self.out_weight = Parameter(rng.normal(0, .02, (d_model, d_model)).astype(np.float32))
        self.out_bias = Parameter(np.zeros(d_model, dtype=np.float32))
        hidden = d_model * ff_mult
        self.ff1_weight = Parameter(rng.normal(0, .02, (d_model, hidden)).astype(np.float32))
        self.ff1_bias = Parameter(np.zeros(hidden, dtype=np.float32))
        self.ff2_weight = Parameter(rng.normal(0, .02, (hidden, d_model)).astype(np.float32))
        self.ff2_bias = Parameter(np.zeros(d_model, dtype=np.float32))

    def forward(self, inputs, attention_mask=None, cache=False):
        batch, length, d_model = inputs.shape
        qkv = inputs @ self.qkv_weight.data + self.qkv_bias.data
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = (value.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
                   for value in (q, k, v))
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -1e9, scores)
        if attention_mask is not None:
            scores = np.where(attention_mask[:, None, None, :] > 0, scores, -1e9)
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        dropped_attention, attention_keep = _dropout(probabilities, self.dropout, self.training)
        context = dropped_attention @ v
        context = context.transpose(0, 2, 1, 3).reshape(batch, length, d_model)
        attention, attention_keep_output = _dropout(
            context @ self.out_weight.data + self.out_bias.data, self.dropout, self.training
        )
        residual = inputs + attention
        ff_pre = residual @ self.ff1_weight.data + self.ff1_bias.data
        ff_hidden = np.maximum(ff_pre, 0)
        ff, ff_keep = _dropout(ff_hidden @ self.ff2_weight.data + self.ff2_bias.data, self.dropout, self.training)
        output = residual + ff
        if not cache:
            return output
        return output, (inputs, q, k, v, probabilities, dropped_attention, attention_keep, context,
                        attention_keep_output, residual, ff_pre, ff_hidden, ff_keep)

    def backward(self, gradient, cache):
        (inputs, q, k, v, probabilities, dropped_attention, attention_keep, context,
         attention_keep_output, residual, ff_pre, ff_hidden, ff_keep) = cache
        gradient_ff = gradient if ff_keep is None else gradient * ff_keep / (1.0 - self.dropout)
        self.ff2_weight.grad[...] = ff_hidden.reshape(-1, ff_hidden.shape[-1]).T @ gradient_ff.reshape(-1, gradient_ff.shape[-1])
        self.ff2_bias.grad[...] = gradient_ff.sum(axis=(0, 1))
        gradient_hidden = gradient_ff @ self.ff2_weight.data.T
        gradient_hidden *= ff_pre > 0
        self.ff1_weight.grad[...] = residual.reshape(-1, residual.shape[-1]).T @ gradient_hidden.reshape(-1, gradient_hidden.shape[-1])
        self.ff1_bias.grad[...] = gradient_hidden.sum(axis=(0, 1))
        gradient_residual = gradient + gradient_hidden @ self.ff1_weight.data.T
        gradient_attention = gradient_residual if attention_keep_output is None else gradient_residual * attention_keep_output / (1.0 - self.dropout)
        gradient_context = gradient_attention @ self.out_weight.data.T
        self.out_weight.grad[...] = context.reshape(-1, context.shape[-1]).T @ gradient_attention.reshape(-1, gradient_attention.shape[-1])
        self.out_bias.grad[...] = gradient_attention.sum(axis=(0, 1))
        gradient_context = gradient_context.reshape(context.shape[0], context.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        gradient_probabilities = gradient_context @ v.transpose(0, 1, 3, 2)
        if attention_keep is not None:
            gradient_probabilities *= attention_keep / (1.0 - self.dropout)
        gradient_v = dropped_attention.transpose(0, 1, 3, 2) @ gradient_context
        gradient_scores = probabilities * (gradient_probabilities - (gradient_probabilities * probabilities).sum(axis=-1, keepdims=True))
        scale = 1.0 / np.sqrt(self.head_dim)
        gradient_q = gradient_scores @ k * scale
        gradient_k = gradient_scores.transpose(0, 1, 3, 2) @ q * scale
        gradient_qkv = np.concatenate((gradient_q, gradient_k, gradient_v), axis=-1).transpose(0, 2, 1, 3).reshape(inputs.shape[0], inputs.shape[1], -1)
        self.qkv_weight.grad[...] = inputs.reshape(-1, inputs.shape[-1]).T @ gradient_qkv.reshape(-1, gradient_qkv.shape[-1])
        self.qkv_bias.grad[...] = gradient_qkv.sum(axis=(0, 1))
        return gradient_residual + gradient_qkv @ self.qkv_weight.data.T


class TinyTransformer(Module):
    """A fast embedding plus pooled/sequence language model."""
    def __init__(self, vocab_size, d_model, layers, heads, ff_mult, dropout, max_seq_len,
                 task="text-generation", n_classes=0, pad_id=0):
        self.training = True
        self.task, self.max_seq_len, self.pad_id = task, max_seq_len, pad_id
        self.d_model = d_model
        rng = np.random.default_rng()
        self.tok_emb = Parameter(rng.normal(0, .02, (vocab_size, d_model)).astype(np.float32))
        self.pos_emb = Parameter(rng.normal(0, .02, (max_seq_len, d_model)).astype(np.float32))
        self.blocks = [TransformerBlock(d_model, heads, ff_mult, dropout) for _ in range(layers)]
        if task == "text-generation":
            self.head_bias = Parameter(np.zeros(vocab_size, dtype=np.float32))
            self.head_weight = self.tok_emb
        elif task == "text-classification":
            self.head_weight = Parameter(rng.normal(0, .02, (d_model, n_classes)).astype(np.float32))
            self.head_bias = Parameter(np.zeros(n_classes, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported task for TinyTransformer: {task}")

    def forward(self, input_ids, attention_mask=None, cache=False):
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise AssertionError(f"Sequence length {length} exceeds max_seq_len {self.max_seq_len}")
        hidden = self.tok_emb.data[input_ids] + self.pos_emb.data[np.arange(length)]
        caches = []
        for block in self.blocks:
            if cache:
                hidden, block_cache = block.forward(hidden, attention_mask, cache=True)
                caches.append(block_cache)
            else:
                hidden = block.forward(hidden, attention_mask)
        if self.task == "text-generation":
            out = hidden @ self.tok_emb.data.T + self.head_bias.data
        else:
            mask = np.ones((batch, length), dtype=np.float32) if attention_mask is None else attention_mask
            pooled = (hidden * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out = pooled @ self.head_weight.data + self.head_bias.data
        return (out, (input_ids, hidden, attention_mask, caches)) if cache else out

    def loss_and_backward(self, input_ids, target, attention_mask=None):
        logits, (ids, hidden, mask, caches) = self.forward(input_ids, attention_mask, cache=True)
        loss, grad = softmax_cross_entropy(logits, target, self.pad_id if self.task == "text-generation" else None)
        for parameter in self.parameters():
            parameter.grad.fill(0)
        self.head_bias.grad[...] = grad.reshape(-1, grad.shape[-1]).sum(axis=0)
        if self.task == "text-generation":
            flat_h, flat_g = hidden.reshape(-1, hidden.shape[-1]), grad.reshape(-1, grad.shape[-1])
            self.tok_emb.grad[...] += flat_g.T @ flat_h
            dh = (flat_g @ self.tok_emb.data).reshape(hidden.shape)
        else:
            valid_mask = np.ones(ids.shape, dtype=np.float32) if mask is None else mask
            pooled = (hidden * valid_mask[:, :, None]).sum(1) / np.maximum(valid_mask.sum(1, keepdims=True), 1)
            self.head_weight.grad[...] = pooled.T @ grad
            self.head_bias.grad[...] = grad.sum(0)
            dh = (grad @ self.head_weight.data.T)[:, None, :] * valid_mask[:, :, None]
            dh /= np.maximum(valid_mask.sum(1, keepdims=True)[:, :, None], 1)
        for block, block_cache in zip(self.blocks[::-1], caches[::-1]):
            dh = block.backward(dh, block_cache)
        self.pos_emb.grad.fill(0)
        np.add.at(self.tok_emb.grad, ids.reshape(-1), dh.reshape(-1, dh.shape[-1]))
        for i in range(ids.shape[1]): self.pos_emb.grad[i] = dh[:, i].sum(0)
        return loss

    def generate(self, input_ids, max_new_tokens, temperature=.8, top_k=40, eos_id: Optional[int] = None):
        ids = np.asarray(input_ids, dtype=np.int64).copy()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.max_seq_len:])[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                excluded = np.argpartition(logits, -k, axis=1)[:, :-k]
                logits[np.arange(len(ids))[:, None], excluded] = -np.inf
            probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
            next_ids = np.array([np.random.choice(logits.shape[1], p=p) for p in probs])[:, None]
            ids = np.concatenate([ids, next_ids], axis=1)
            if eos_id is not None and np.all(next_ids == eos_id): break
        return ids
        """A small, dependency-free legacy implementation retained only as inert text.

Used for:
  - "text-generation": next-token prediction over the char vocabulary
  - "text-classification": same backbone, with a classification head on
    the final token's hidden state instead of a language-modeling head

Kept intentionally compact -- this is not meant to compete with
production LLM training frameworks, it's meant to give Tensorless a real,
working, from-scratch model that trains fast enough on CPU for the
"zero setup" experience to actually be pleasant.

from __future__ import annotations

import math
from typing import Optional

import numpy as np


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

    def forward(self, x: np.ndarray, attn_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_mult, dropout)

    def forward(self, x: np.ndarray, attn_mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    Legacy decoder-only transformer usable for LM or sequence classification.

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
        input_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
    ) -> np.ndarray:
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
    """
