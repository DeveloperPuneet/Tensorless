"""MLX implementation of the text transformer.

MLX is imported lazily so Tensorless remains usable on non-Apple systems
without installing an accelerator framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.transformer import TinyTransformer


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError("MLX is required for the mps backend; install tensorless[mps].") from exc
    return mx


def is_available() -> bool:
    try:
        _mlx()
        return True
    except ImportError:
        return False


class MlxTinyTransformer(TinyTransformer):
    """The native Tensorless parameter layout with MLX math and gradients."""

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", "fp32")
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", args[1] if len(args) > 1 else None)
        self.heads = kwargs.get("heads", args[3] if len(args) > 3 else None)
        if d_model is None or self.heads is None:
            raise ValueError("MLX transformer requires d_model and heads")
        self.d_model = d_model
        self.head_dim = d_model // self.heads

    def _mlx_params(self):
        mx = _mlx()
        dtype = {"fp16": mx.float16, "bf16": mx.bfloat16}.get(self.precision, mx.float32)
        return {name: mx.array(parameter.data, dtype=dtype) for name, parameter in self.named_parameters()}

    def _forward_mlx(self, params, input_ids, attention_mask=None):
        mx = _mlx()
        input_ids = mx.array(input_ids, dtype=mx.int32)
        batch, length = input_ids.shape
        hidden = params["tok_emb"][input_ids] + params["pos_emb"][mx.arange(length)]
        causal = mx.tril(mx.ones((length, length), dtype=mx.bool_))
        for index in range(len(self.blocks)):
            prefix = f"blocks.{index}"
            qkv = hidden @ params[f"{prefix}.qkv_weight"] + params[f"{prefix}.qkv_bias"]
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = q.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
            scores = mx.where(causal[None, None, :, :], scores, -1e9)
            if attention_mask is not None:
                mask = mx.array(attention_mask) > 0
                scores = mx.where(mask[:, None, None, :], scores, -1e9)
            probabilities = mx.softmax(scores, axis=-1)
            context = probabilities @ v
            context = context.transpose(0, 2, 1, 3).reshape(batch, length, self.d_model)
            attention = context @ params[f"{prefix}.out_weight"] + params[f"{prefix}.out_bias"]
            residual = hidden + attention
            ff_pre = residual @ params[f"{prefix}.ff1_weight"] + params[f"{prefix}.ff1_bias"]
            ff_hidden = mx.maximum(ff_pre, 0)
            ff = ff_hidden @ params[f"{prefix}.ff2_weight"] + params[f"{prefix}.ff2_bias"]
            hidden = residual + ff
        if self.task == "text-generation":
            return hidden @ params["tok_emb"].T + params["head_bias"]
        mask = mx.ones((batch, length)) if attention_mask is None else mx.array(attention_mask)
        pooled = (hidden * mask[:, :, None]).sum(1) / mx.maximum(mask.sum(1, keepdims=True), 1)
        return pooled @ params["head_weight"] + params["head_bias"]

    def forward(self, input_ids, attention_mask=None, cache=False):
        output = np.asarray(self._forward_mlx(self._mlx_params(), input_ids, attention_mask))
        return (output, None) if cache else output

    def loss_and_backward(self, input_ids, target, attention_mask=None):
        mx = _mlx()
        target = mx.array(target, dtype=mx.int32)
        params = self._mlx_params()

        def loss_fn(current):
            logits = self._forward_mlx(current, input_ids, attention_mask)
            flat_logits = logits.reshape((-1, logits.shape[-1])).astype(mx.float32)
            flat_target = target.reshape((-1,))
            log_probs = flat_logits - mx.logsumexp(flat_logits, axis=-1, keepdims=True)
            losses = -mx.take_along_axis(log_probs, flat_target[:, None], axis=1).squeeze(1)
            if self.task == "text-generation":
                valid = flat_target != self.pad_id
                return mx.sum(mx.where(valid, losses, 0.0)) / mx.maximum(mx.sum(valid), 1)
            return mx.mean(losses)

        loss, gradients = mx.value_and_grad(loss_fn)(params)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32)
        return float(np.asarray(loss))

    def generate(self, input_ids, max_new_tokens, temperature=.8, top_k=40, eos_id: Optional[int] = None):
        ids = np.asarray(input_ids, dtype=np.int64).copy()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.max_seq_len:])[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                excluded = np.argpartition(logits, -k, axis=1)[:, :-k]
                logits[np.arange(len(ids))[:, None], excluded] = -np.inf
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            next_ids = np.array([np.random.choice(logits.shape[1], p=p) for p in probs])[:, None]
            ids = np.concatenate([ids, next_ids], axis=1)
            if eos_id is not None and np.all(next_ids == eos_id):
                break
        return ids
