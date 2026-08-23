"""MLX implementation of the text transformer.

MLX is imported lazily so Tensorless remains usable on non-Apple systems
without installing an accelerator framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.transformer import TinyTransformer
from ..models.mlp import TabularMLP


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

        def layer_norm(x, gain, bias, eps=1e-5):
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.var(x, axis=-1, keepdims=True)
            return (x - mean) / mx.sqrt(var + eps) * gain + bias

        input_ids = mx.array(input_ids, dtype=mx.int32)
        batch, length = input_ids.shape
        hidden = params["tok_emb"][input_ids] + params["pos_emb"][mx.arange(length)]
        causal = mx.tril(mx.ones((length, length), dtype=mx.bool_))
        for index in range(len(self.blocks)):
            prefix = f"blocks.{index}"
            normed1 = layer_norm(hidden, params[f"{prefix}.ln1_gain"], params[f"{prefix}.ln1_bias"])
            qkv = normed1 @ params[f"{prefix}.qkv_weight"] + params[f"{prefix}.qkv_bias"]
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
            normed2 = layer_norm(residual, params[f"{prefix}.ln2_gain"], params[f"{prefix}.ln2_bias"])
            ff_pre = normed2 @ params[f"{prefix}.ff1_weight"] + params[f"{prefix}.ff1_bias"]
            ff_hidden = mx.maximum(ff_pre, 0)
            ff = ff_hidden @ params[f"{prefix}.ff2_weight"] + params[f"{prefix}.ff2_bias"]
            hidden = residual + ff
        hidden = layer_norm(hidden, params["ln_f_gain"], params["ln_f_bias"])
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

        loss_scale = 128.0 if self.precision == "fp16" else 1.0
        scaled_loss_fn = lambda current: loss_fn(current) * loss_scale
        loss, gradients = mx.value_and_grad(scaled_loss_fn)(params)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32) / loss_scale
        return float(np.asarray(loss)) / loss_scale

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


def _mlx_gelu(x, mx):
    scale = float(np.sqrt(2.0 / np.pi))
    scaled = scale * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + mx.tanh(scaled))


class MlxTabularMLP(TabularMLP):
    """The native tabular-MLP parameter layout with MLX math and gradients.

    Used automatically instead of the plain NumPy `TabularMLP` whenever the
    resolved device is `mps`, so tabular training actually runs on Apple
    Silicon's GPU instead of silently staying on CPU.
    """

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", "fp32")
        super().__init__(*args, **kwargs)

    def _mlx_params(self):
        mx = _mlx()
        dtype = {"fp16": mx.float16, "bf16": mx.bfloat16}.get(self.precision, mx.float32)
        return {name: mx.array(parameter.data, dtype=dtype) for name, parameter in self.named_parameters()}

    def _forward_mlx(self, params, numeric, categorical, dropout_masks=None):
        mx = _mlx()
        parts = [mx.array(numeric, dtype=mx.float32)] if self.n_numeric else []
        categorical = mx.array(categorical, dtype=mx.int32)
        for i in range(len(self.embeddings)):
            parts.append(params[f"embeddings.{i}"][categorical[:, i]])
        x = mx.concatenate(parts, axis=1) if parts else mx.array(numeric, dtype=mx.float32)
        for i in range(len(self.weights)):
            preactivation = x @ params[f"weights.{i}"] + params[f"biases.{i}"]
            x = _mlx_gelu(preactivation, mx)
            if dropout_masks is not None and dropout_masks[i] is not None:
                x = x * dropout_masks[i] / (1.0 - self.dropout)
        out = x @ params["head_weight"] + params["head_bias"]
        if self.task == "regression":
            out = out[:, 0]
        return out

    def _make_dropout_masks(self, batch_size):
        if not self.training or self.dropout <= 0:
            return None
        mx = _mlx()
        return [
            mx.array((np.random.random((batch_size, weight.data.shape[1])) >= self.dropout).astype(np.float32))
            for weight in self.weights
        ]

    def forward(self, numeric, categorical, cache=False):
        params = self._mlx_params()
        batch_size = numeric.shape[0] if self.n_numeric else categorical.shape[0]
        dropout_masks = self._make_dropout_masks(batch_size)
        out = np.asarray(self._forward_mlx(params, numeric, categorical, dropout_masks))
        return (out, None) if cache else out

    def loss_and_backward(self, numeric, categorical, target):
        mx = _mlx()
        params = self._mlx_params()
        batch_size = numeric.shape[0] if self.n_numeric else categorical.shape[0]
        dropout_masks = self._make_dropout_masks(batch_size)
        is_regression = self.task == "regression"
        target_arr = mx.array(target, dtype=mx.float32 if is_regression else mx.int32)

        def loss_fn(current):
            logits = self._forward_mlx(current, numeric, categorical, dropout_masks)
            if is_regression:
                error = logits - target_arr
                return mx.mean(error * error)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.take_along_axis(log_probs, target_arr[:, None], axis=1))

        loss, gradients = mx.value_and_grad(loss_fn)(params)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32)
        return float(np.asarray(loss))
