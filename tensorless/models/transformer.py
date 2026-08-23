"""Compact NumPy text model retaining Tensorless transformer behavior.

Architecture: a standard pre-norm decoder-only transformer -- LayerNorm
before attention and before the feed-forward block, plus a final LayerNorm
before the output head. Normalization is what keeps deep stacks of residual
blocks trainable; without it, gradients and activations drift as layers are
stacked and generation quality suffers noticeably once you go beyond one or
two blocks.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from ..engine import Module, Parameter, softmax_cross_entropy, layer_norm_forward, layer_norm_backward


def _dropout(value, probability, training):
    if not training or probability <= 0:
        return value, None
    keep = np.random.random(value.shape) >= probability
    return value * keep / (1.0 - probability), keep


def _gelu(value):
    scaled = np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)
    return 0.5 * value * (1.0 + np.tanh(scaled))


def _gelu_backward(value):
    scaled = np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)
    tanh_scaled = np.tanh(scaled)
    return 0.5 * (1.0 + tanh_scaled) + 0.5 * value * (1.0 - tanh_scaled ** 2) * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * value ** 2)


class TransformerBlock(Module):
    def __init__(self, d_model, heads, ff_mult, dropout):
        if d_model % heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads, self.head_dim, self.dropout = heads, d_model // heads, dropout
        self.ln1_gain = Parameter(np.ones(d_model, dtype=np.float32))
        self.ln1_bias = Parameter(np.zeros(d_model, dtype=np.float32))
        self.qkv_weight = Parameter(np.random.normal(0, .02, (d_model, 3 * d_model)).astype(np.float32))
        self.qkv_bias = Parameter(np.zeros(3 * d_model, dtype=np.float32))
        self.out_weight = Parameter(np.random.normal(0, .02, (d_model, d_model)).astype(np.float32))
        self.out_bias = Parameter(np.zeros(d_model, dtype=np.float32))
        self.ln2_gain = Parameter(np.ones(d_model, dtype=np.float32))
        self.ln2_bias = Parameter(np.zeros(d_model, dtype=np.float32))
        hidden = d_model * ff_mult
        self.ff1_weight = Parameter(np.random.normal(0, .02, (d_model, hidden)).astype(np.float32))
        self.ff1_bias = Parameter(np.zeros(hidden, dtype=np.float32))
        self.ff2_weight = Parameter(np.random.normal(0, .02, (hidden, d_model)).astype(np.float32))
        self.ff2_bias = Parameter(np.zeros(d_model, dtype=np.float32))

    def forward(self, inputs, attention_mask=None, cache=False):
        batch, length, d_model = inputs.shape
        normed1, ln1_cache = layer_norm_forward(inputs, self.ln1_gain.data, self.ln1_bias.data)
        qkv = normed1 @ self.qkv_weight.data + self.qkv_bias.data
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
        normed2, ln2_cache = layer_norm_forward(residual, self.ln2_gain.data, self.ln2_bias.data)
        ff_pre = normed2 @ self.ff1_weight.data + self.ff1_bias.data
        ff_hidden = _gelu(ff_pre)
        ff, ff_keep = _dropout(ff_hidden @ self.ff2_weight.data + self.ff2_bias.data, self.dropout, self.training)
        output = residual + ff
        if not cache:
            return output
        return output, (inputs, normed1, ln1_cache, q, k, v, probabilities, dropped_attention, attention_keep,
                        context, attention_keep_output, residual, normed2, ln2_cache, ff_pre, ff_hidden, ff_keep)

    def backward(self, gradient, cache):
        (inputs, normed1, ln1_cache, q, k, v, probabilities, dropped_attention, attention_keep, context,
         attention_keep_output, residual, normed2, ln2_cache, ff_pre, ff_hidden, ff_keep) = cache
        gradient_ff = gradient if ff_keep is None else gradient * ff_keep / (1.0 - self.dropout)
        self.ff2_weight.grad[...] = ff_hidden.reshape(-1, ff_hidden.shape[-1]).T @ gradient_ff.reshape(-1, gradient_ff.shape[-1])
        self.ff2_bias.grad[...] = gradient_ff.sum(axis=(0, 1))
        gradient_hidden = gradient_ff @ self.ff2_weight.data.T
        gradient_hidden *= _gelu_backward(ff_pre)
        self.ff1_weight.grad[...] = normed2.reshape(-1, normed2.shape[-1]).T @ gradient_hidden.reshape(-1, gradient_hidden.shape[-1])
        self.ff1_bias.grad[...] = gradient_hidden.sum(axis=(0, 1))
        gradient_normed2 = gradient_hidden @ self.ff1_weight.data.T
        gradient_residual_from_ff, ln2_gain_grad, ln2_bias_grad = layer_norm_backward(gradient_normed2, ln2_cache)
        self.ln2_gain.grad[...] = ln2_gain_grad
        self.ln2_bias.grad[...] = ln2_bias_grad
        gradient_residual = gradient + gradient_residual_from_ff
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
        # Merge each of q/k/v back from (batch, heads, length, head_dim) to
        # (batch, length, d_model) *before* concatenating them, matching the
        # forward pass's layout ([q_block | k_block | v_block] along the last
        # axis, each block itself head-major). Concatenating on the head-dim
        # axis first (as a prior version of this code did) interleaves heads
        # and q/k/v in the wrong order and silently corrupts the qkv_weight /
        # qkv_bias gradients for any heads > 1 configuration.
        batch, length = inputs.shape[0], inputs.shape[1]

        def _merge_heads(gradient):
            return gradient.transpose(0, 2, 1, 3).reshape(batch, length, -1)

        gradient_qkv = np.concatenate(
            (_merge_heads(gradient_q), _merge_heads(gradient_k), _merge_heads(gradient_v)), axis=-1
        )
        self.qkv_weight.grad[...] = normed1.reshape(-1, normed1.shape[-1]).T @ gradient_qkv.reshape(-1, gradient_qkv.shape[-1])
        self.qkv_bias.grad[...] = gradient_qkv.sum(axis=(0, 1))
        gradient_normed1 = gradient_qkv @ self.qkv_weight.data.T
        gradient_inputs_from_attn, ln1_gain_grad, ln1_bias_grad = layer_norm_backward(gradient_normed1, ln1_cache)
        self.ln1_gain.grad[...] = ln1_gain_grad
        self.ln1_bias.grad[...] = ln1_bias_grad
        return gradient_residual + gradient_inputs_from_attn


class TinyTransformer(Module):
    """A fast embedding plus pooled/sequence language model."""
    def __init__(self, vocab_size, d_model, layers, heads, ff_mult, dropout, max_seq_len,
                 task="text-generation", n_classes=0, pad_id=0, gradient_checkpointing=False):
        self.training = True
        self.task, self.max_seq_len, self.pad_id = task, max_seq_len, pad_id
        self.gradient_checkpointing = gradient_checkpointing
        self.d_model = d_model
        self.tok_emb = Parameter(np.random.normal(0, .02, (vocab_size, d_model)).astype(np.float32))
        self.pos_emb = Parameter(np.random.normal(0, .02, (max_seq_len, d_model)).astype(np.float32))
        self.blocks = [TransformerBlock(d_model, heads, ff_mult, dropout) for _ in range(layers)]
        self.ln_f_gain = Parameter(np.ones(d_model, dtype=np.float32))
        self.ln_f_bias = Parameter(np.zeros(d_model, dtype=np.float32))
        if task == "text-generation":
            self.head_bias = Parameter(np.zeros(vocab_size, dtype=np.float32))
            self.head_weight = self.tok_emb
        elif task == "text-classification":
            self.head_weight = Parameter(np.random.normal(0, .02, (d_model, n_classes)).astype(np.float32))
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
                if self.gradient_checkpointing:
                    rng_state = np.random.get_state()
                    block_input = hidden.copy()
                    hidden = block.forward(hidden, attention_mask)
                    caches.append((block_input, attention_mask, rng_state))
                else:
                    hidden, block_cache = block.forward(hidden, attention_mask, cache=True)
                    caches.append(block_cache)
            else:
                hidden = block.forward(hidden, attention_mask)
        hidden_normed, lnf_cache = layer_norm_forward(hidden, self.ln_f_gain.data, self.ln_f_bias.data)
        if self.task == "text-generation":
            out = hidden_normed @ self.tok_emb.data.T + self.head_bias.data
        else:
            mask = np.ones((batch, length), dtype=np.float32) if attention_mask is None else attention_mask
            pooled = (hidden_normed * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out = pooled @ self.head_weight.data + self.head_bias.data
        return (out, (input_ids, hidden, hidden_normed, lnf_cache, attention_mask, caches)) if cache else out

    def loss_and_backward(self, input_ids, target, attention_mask=None):
        logits, (ids, hidden, hidden_normed, lnf_cache, mask, caches) = self.forward(input_ids, attention_mask, cache=True)
        loss, grad = softmax_cross_entropy(logits, target, self.pad_id if self.task == "text-generation" else None)
        for parameter in self.parameters():
            parameter.grad.fill(0)
        self.head_bias.grad[...] = grad.reshape(-1, grad.shape[-1]).sum(axis=0)
        if self.task == "text-generation":
            flat_hn, flat_g = hidden_normed.reshape(-1, hidden_normed.shape[-1]), grad.reshape(-1, grad.shape[-1])
            self.tok_emb.grad[...] += flat_g.T @ flat_hn
            d_hidden_normed = (flat_g @ self.tok_emb.data).reshape(hidden_normed.shape)
        else:
            valid_mask = np.ones(ids.shape, dtype=np.float32) if mask is None else mask
            pooled = (hidden_normed * valid_mask[:, :, None]).sum(1) / np.maximum(valid_mask.sum(1, keepdims=True), 1)
            self.head_weight.grad[...] = pooled.T @ grad
            self.head_bias.grad[...] = grad.sum(0)
            d_hidden_normed = (grad @ self.head_weight.data.T)[:, None, :] * valid_mask[:, :, None]
            d_hidden_normed /= np.maximum(valid_mask.sum(1, keepdims=True)[:, :, None], 1)
        dh, lnf_gain_grad, lnf_bias_grad = layer_norm_backward(d_hidden_normed, lnf_cache)
        self.ln_f_gain.grad[...] = lnf_gain_grad
        self.ln_f_bias.grad[...] = lnf_bias_grad
        for block, block_cache in zip(self.blocks[::-1], caches[::-1]):
            if self.gradient_checkpointing:
                block_input, block_mask, rng_state = block_cache
                current_rng_state = np.random.get_state()
                np.random.set_state(rng_state)
                _, block_cache = block.forward(block_input, block_mask, cache=True)
                np.random.set_state(current_rng_state)
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
