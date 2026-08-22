"""JAX implementation of the text transformer.

JAX is imported lazily so importing Tensorless remains CPU/NumPy-only when the
optional dependency is not installed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.transformer import TinyTransformer


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX is required for the cuda/tpu backend; install tensorless[jax].") from exc
    return jax, jnp


def is_available() -> bool:
    try:
        _jax()
        return True
    except ImportError:
        return False


class JaxTinyTransformer(TinyTransformer):
    """The native transformer parameter layout with JAX math and gradients."""

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", "fp32")
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", args[1] if len(args) > 1 else None)
        self.heads = kwargs.get("heads", args[3] if len(args) > 3 else None)
        if d_model is None or self.heads is None:
            raise ValueError("JAX transformer requires d_model and heads")
        self.d_model = d_model
        self.head_dim = d_model // self.heads

    def _jax_params(self):
        _, jnp = _jax()
        dtype = {"fp16": jnp.float16, "bf16": jnp.bfloat16}.get(self.precision, jnp.float32)
        return {name: jnp.asarray(value.data, dtype=dtype) for name, value in self.named_parameters()}

    def _forward_jax(self, params, input_ids, attention_mask=None):
        _, jnp = _jax()
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        batch, length = input_ids.shape
        hidden = params["tok_emb"][input_ids] + params["pos_emb"][jnp.arange(length)]
        causal = jnp.tril(jnp.ones((length, length), dtype=bool))
        for index in range(len(self.blocks)):
            prefix = f"blocks.{index}"
            qkv = hidden @ params[f"{prefix}.qkv_weight"] + params[f"{prefix}.qkv_bias"]
            q, k, v = jnp.split(qkv, 3, axis=-1)
            q = q.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
            scores = jnp.where(causal[None, None, :, :], scores, -1e9)
            if attention_mask is not None:
                mask = jnp.asarray(attention_mask) > 0
                scores = jnp.where(mask[:, None, None, :], scores, -1e9)
            probabilities = jnp.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            context = probabilities @ v
            context = context.transpose(0, 2, 1, 3).reshape(batch, length, self.d_model)
            attention = context @ params[f"{prefix}.out_weight"] + params[f"{prefix}.out_bias"]
            residual = hidden + attention
            ff_pre = residual @ params[f"{prefix}.ff1_weight"] + params[f"{prefix}.ff1_bias"]
            ff_hidden = jnp.maximum(ff_pre, 0)
            ff = ff_hidden @ params[f"{prefix}.ff2_weight"] + params[f"{prefix}.ff2_bias"]
            hidden = residual + ff
        if self.task == "text-generation":
            return hidden @ params["tok_emb"].T + params["head_bias"]
        mask = jnp.ones((batch, length), dtype=jnp.float32) if attention_mask is None else jnp.asarray(attention_mask)
        pooled = (hidden * mask[:, :, None]).sum(1) / jnp.maximum(mask.sum(1, keepdims=True), 1)
        return pooled @ params["head_weight"] + params["head_bias"]

    def forward(self, input_ids, attention_mask=None, cache=False):
        _, jnp = _jax()
        output = self._forward_jax(self._jax_params(), input_ids, attention_mask)
        output = np.asarray(output)
        return (output, None) if cache else output

    def loss_and_backward(self, input_ids, target, attention_mask=None):
        jax, jnp = _jax()
        params = self._jax_params()
        target = jnp.asarray(target, dtype=jnp.int32)

        def loss_fn(current):
            logits = self._forward_jax(current, input_ids, attention_mask)
            flat_logits = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
            flat_target = target.reshape(-1)
            log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
            losses = -jnp.take_along_axis(log_probs, flat_target[:, None], axis=1).squeeze(1)
            if self.task == "text-generation":
                valid = flat_target != self.pad_id
                return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)
            return jnp.mean(losses)

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32)
        return float(loss)

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
