"""JAX implementation of the text transformer.

JAX is imported lazily so importing Tensorless remains CPU/NumPy-only when the
optional dependency is not installed.
"""

from __future__ import annotations

from typing import Optional
import os

import numpy as np

from ..models.transformer import TinyTransformer
from ..models.mlp import TabularMLP


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        distributed_vars = ("JAX_COORDINATOR_ADDRESS", "JAX_PROCESS_COUNT", "JAX_PROCESS_ID")
        if all(name in os.environ for name in distributed_vars) and not jax.distributed.is_initialized():
            jax.distributed.initialize(
                coordinator_address=os.environ["JAX_COORDINATOR_ADDRESS"],
                num_processes=int(os.environ["JAX_PROCESS_COUNT"]),
                process_id=int(os.environ["JAX_PROCESS_ID"]),
            )
    except ImportError as exc:
        raise ImportError("JAX is required for the cuda/tpu backend; install tensorless[jax].") from exc
    return jax, jnp


def is_available() -> bool:
    try:
        _jax()
        return True
    except ImportError:
        return False


def supports_fused_training(model) -> bool:
    """Whether `model` can use the device-resident fused AdamW/Adam train
    step (see `_JaxFusedAdamMixin`) instead of round-tripping full
    parameter/gradient dicts through host NumPy every step."""
    return isinstance(model, _JaxFusedAdamMixin)


def _adam_update_tree(jnp, params, m, v, grads, step_count, lr, weight_decay, decoupled):
    """Fused AdamW/Adam parameter update over a pytree dict of device arrays.

    Runs entirely inside `jax.jit` as part of the fused train step below, so
    params/moments never have to leave the accelerator to be updated.
    """
    beta1_correction = 1.0 - 0.9 ** step_count
    beta2_correction = 1.0 - 0.999 ** step_count
    new_params, new_m, new_v = {}, {}, {}
    for name, p in params.items():
        g = grads[name]
        if decoupled:
            p = p * (1.0 - lr * weight_decay)
        else:
            g = g + weight_decay * p
        m_i = 0.9 * m[name] + 0.1 * g
        v_i = 0.999 * v[name] + 0.001 * g * g
        m_hat = m_i / beta1_correction
        v_hat = v_i / beta2_correction
        new_params[name] = p - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        new_m[name] = m_i
        new_v[name] = v_i
    return new_params, new_m, new_v


class _JaxFusedAdamMixin:
    """Keeps parameters + Adam moment estimates resident on the GPU/TPU
    across many training steps, instead of re-uploading the full parameter
    dict and downloading the full gradient dict through host NumPy on
    *every single step*.

    Compiling the forward pass with `jax.jit` (see `_get_forward_jit`) fixes
    re-tracing overhead, but each `loss_and_backward()` call still does two
    full host<->device transfers of the whole model (params up, grads down),
    and the NumPy `Adam`/`AdamW` optimizer in `engine.py` then runs the
    update on the CPU. For small/medium models that transfer, not compute,
    dominates the step time and is why GPU utilization can stay low even
    after JIT is fixed. This mixin fuses forward+backward+optimizer-update
    into one jitted function so state only crosses the host/device boundary
    at explicit sync points (checkpointing, validation, end of training) --
    see `training/trainer.py`'s use of `init_fused_state`/`sync_fused_to_host`.
    """

    _fused_params = None
    _fused_m = None
    _fused_v = None
    _fused_step = 0
    _fused_train_step_key = None

    def fused_training_supported(self) -> bool:
        return True

    def init_fused_state(self, weight_decay: float, decoupled: bool, step: int = 0):
        _, jnp = _jax()
        params = self._jax_params()
        self._fused_params = params
        self._fused_m = {name: jnp.zeros_like(value) for name, value in params.items()}
        self._fused_v = {name: jnp.zeros_like(value) for name, value in params.items()}
        self._fused_step = int(step)
        self._fused_train_step_key = None

    def load_fused_optimizer_moments(self, m_by_name, v_by_name):
        """Restore Adam moment estimates (as NumPy, from a resumed checkpoint)."""
        _, jnp = _jax()
        self._fused_m = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in m_by_name.items()}
        self._fused_v = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in v_by_name.items()}

    def sync_fused_to_host(self):
        """Write device-resident params/moments back to host NumPy.

        Called only at checkpoint/validation/end-of-training boundaries, so
        this cost is paid a handful of times per epoch rather than every step.
        """
        if self._fused_params is None:
            return {}, {}
        for name, parameter in self.named_parameters():
            parameter.data[...] = np.asarray(self._fused_params[name], dtype=np.float32)
        m_by_name = {name: np.asarray(value, dtype=np.float32) for name, value in self._fused_m.items()}
        v_by_name = {name: np.asarray(value, dtype=np.float32) for name, value in self._fused_v.items()}
        return m_by_name, v_by_name



class JaxTinyTransformer(_JaxFusedAdamMixin, TinyTransformer):
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

        def layer_norm(x, gain, bias, eps=1e-5):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + eps) * gain + bias

        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        batch, length = input_ids.shape
        hidden = params["tok_emb"][input_ids] + params["pos_emb"][jnp.arange(length)]
        causal = jnp.tril(jnp.ones((length, length), dtype=bool))
        for index in range(len(self.blocks)):
            prefix = f"blocks.{index}"
            normed1 = layer_norm(hidden, params[f"{prefix}.ln1_gain"], params[f"{prefix}.ln1_bias"])
            qkv = normed1 @ params[f"{prefix}.qkv_weight"] + params[f"{prefix}.qkv_bias"]
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
            normed2 = layer_norm(residual, params[f"{prefix}.ln2_gain"], params[f"{prefix}.ln2_bias"])
            ff_pre = normed2 @ params[f"{prefix}.ff1_weight"] + params[f"{prefix}.ff1_bias"]
            ff_hidden = jnp.maximum(ff_pre, 0)
            ff = ff_hidden @ params[f"{prefix}.ff2_weight"] + params[f"{prefix}.ff2_bias"]
            hidden = residual + ff
        hidden = layer_norm(hidden, params["ln_f_gain"], params["ln_f_bias"])
        if self.task == "text-generation":
            return hidden @ params["tok_emb"].T + params["head_bias"]
        mask = jnp.ones((batch, length), dtype=jnp.float32) if attention_mask is None else jnp.asarray(attention_mask)
        pooled = (hidden * mask[:, :, None]).sum(1) / jnp.maximum(mask.sum(1, keepdims=True), 1)
        return pooled @ params["head_weight"] + params["head_bias"]

    def forward(self, input_ids, attention_mask=None, cache=False):
        params = self._jax_params()
        output = self._get_forward_jit()(params, input_ids, attention_mask)
        output = np.asarray(output)
        return (output, None) if cache else output

    def _get_forward_jit(self):
        # Compile once per instance and reuse. Without this, every call
        # re-traces the whole forward pass in Python and dispatches ops to
        # the accelerator one at a time -- which shows up as pegged CPU
        # (tracing/dispatch overhead) with near-zero GPU utilization, even
        # though the correct device was selected.
        if getattr(self, "_forward_jit_fn", None) is None:
            jax, _ = _jax()
            self._forward_jit_fn = jax.jit(self._forward_jax)
        return self._forward_jit_fn

    def _get_loss_and_grad_jit(self):
        if getattr(self, "_loss_and_grad_jit_fn", None) is None:
            jax, jnp = _jax()

            def loss_fn(current, current_ids, current_target, current_mask):
                forward = jax.checkpoint(self._forward_jax) if self.gradient_checkpointing else self._forward_jax
                logits = forward(current, current_ids, current_mask)
                flat_logits = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
                flat_target = current_target.reshape(-1)
                log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
                losses = -jnp.take_along_axis(log_probs, flat_target[:, None], axis=1).squeeze(1)
                if self.task == "text-generation":
                    valid = flat_target != self.pad_id
                    return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)
                return jnp.mean(losses)

            loss_scale = 128.0 if self.precision == "fp16" else 1.0
            scaled_loss_fn = lambda *arguments: loss_fn(*arguments) * loss_scale
            self._loss_and_grad_jit_fn = jax.jit(jax.value_and_grad(scaled_loss_fn))
        return self._loss_and_grad_jit_fn

    def _get_pmap_step(self):
        if getattr(self, "_pmap_step_fn", None) is None:
            jax, jnp = _jax()

            def loss_fn(current, current_ids, current_target, current_mask):
                forward = jax.checkpoint(self._forward_jax) if self.gradient_checkpointing else self._forward_jax
                logits = forward(current, current_ids, current_mask)
                flat_logits = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
                flat_target = current_target.reshape(-1)
                log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
                losses = -jnp.take_along_axis(log_probs, flat_target[:, None], axis=1).squeeze(1)
                if self.task == "text-generation":
                    valid = flat_target != self.pad_id
                    return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)
                return jnp.mean(losses)

            loss_scale = 128.0 if self.precision == "fp16" else 1.0
            scaled_loss_fn = lambda *arguments: loss_fn(*arguments) * loss_scale
            per_device_loss = jax.value_and_grad(scaled_loss_fn)

            def mapped_step(current, ids, labels, current_mask):
                value, gradients = per_device_loss(current, ids, labels, current_mask)
                return jax.lax.pmean(value, "data"), jax.tree_util.tree_map(
                    lambda gradient: jax.lax.pmean(gradient, "data"), gradients
                )

            self._pmap_step_fn = jax.pmap(mapped_step, axis_name="data", in_axes=(None, 0, 0, 0))
        return self._pmap_step_fn

    def loss_and_backward(self, input_ids, target, attention_mask=None):
        jax, jnp = _jax()
        params = self._jax_params()
        target = jnp.asarray(target, dtype=jnp.int32)
        loss_scale = 128.0 if self.precision == "fp16" else 1.0
        mask = jnp.ones_like(input_ids, dtype=jnp.float32) if attention_mask is None else jnp.asarray(attention_mask)
        devices = jax.local_device_count()
        if devices > 1 and input_ids.shape[0] % devices == 0:
            shard = lambda value: value.reshape((devices, value.shape[0] // devices) + value.shape[1:])
            loss, gradients = self._get_pmap_step()(
                params, shard(jnp.asarray(input_ids)), shard(target), shard(mask)
            )
            loss = loss[0]
            gradients = jax.tree_util.tree_map(lambda gradient: gradient[0] / loss_scale, gradients)
        else:
            loss, gradients = self._get_loss_and_grad_jit()(params, input_ids, target, mask)
            gradients = jax.tree_util.tree_map(lambda gradient: gradient / loss_scale, gradients)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32)
        return float(loss) / loss_scale

    def _get_fused_train_step(self, weight_decay: float, decoupled: bool):
        cache_key = (weight_decay, decoupled)
        if self._fused_train_step_key != cache_key:
            jax, jnp = _jax()
            loss_scale = 128.0 if self.precision == "fp16" else 1.0

            def loss_fn(params, ids, target, mask):
                forward = jax.checkpoint(self._forward_jax) if self.gradient_checkpointing else self._forward_jax
                logits = forward(params, ids, mask)
                flat_logits = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
                flat_target = target.reshape(-1)
                log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
                losses = -jnp.take_along_axis(log_probs, flat_target[:, None], axis=1).squeeze(1)
                if self.task == "text-generation":
                    valid = flat_target != self.pad_id
                    loss = jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(valid.sum(), 1)
                else:
                    loss = jnp.mean(losses)
                return loss * loss_scale

            grad_fn = jax.value_and_grad(loss_fn)

            def step(params, m, v, step_count, ids, target, mask, lr, grad_clip):
                loss, grads = grad_fn(params, ids, target, mask)
                grads = {name: g / loss_scale for name, g in grads.items()}
                grad_norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grads.values()))
                clip_scale = jnp.where(grad_clip > 0, jnp.minimum(1.0, grad_clip / (grad_norm + 1e-12)), 1.0)
                grads = {name: g * clip_scale for name, g in grads.items()}
                new_params, new_m, new_v = _adam_update_tree(
                    jnp, params, m, v, grads, step_count, lr, weight_decay, decoupled
                )
                return new_params, new_m, new_v, loss / loss_scale, grad_norm

            self._fused_train_step_fn = jax.jit(step)
            self._fused_train_step_key = cache_key
        return self._fused_train_step_fn

    def fused_train_step(self, input_ids, target, attention_mask, lr, weight_decay, grad_clip, decoupled):
        """One fused forward+backward+AdamW step, entirely on-device.

        Call `init_fused_state()` once before the first step and
        `sync_fused_to_host()` before reading `state_dict()`/`forward()` on
        the host (e.g. for validation or checkpointing).
        """
        if self._fused_params is None:
            raise RuntimeError("init_fused_state() must be called before fused_train_step().")
        jax, jnp = _jax()
        step_fn = self._get_fused_train_step(weight_decay, decoupled)
        self._fused_step += 1
        ids_arr = jnp.asarray(input_ids, dtype=jnp.int32)
        target_arr = jnp.asarray(target, dtype=jnp.int32)
        mask_arr = jnp.ones_like(ids_arr, dtype=jnp.float32) if attention_mask is None else jnp.asarray(attention_mask, dtype=jnp.float32)
        new_params, new_m, new_v, loss, grad_norm = step_fn(
            self._fused_params, self._fused_m, self._fused_v,
            jnp.asarray(self._fused_step, dtype=jnp.float32),
            ids_arr, target_arr, mask_arr,
            jnp.asarray(lr, dtype=jnp.float32), jnp.asarray(grad_clip, dtype=jnp.float32),
        )
        self._fused_params, self._fused_m, self._fused_v = new_params, new_m, new_v
        return float(loss), float(grad_norm)

    def generate(self, input_ids, max_new_tokens, temperature=.8, top_k=40, top_p=.9,
                 repetition_penalty=1.1, eos_id: Optional[int] = None):
        # Mirrors `TinyTransformer.generate` (the NumPy path) exactly, so
        # generation quality doesn't quietly regress just because a model
        # happens to run on cuda/tpu. Sampling itself stays in NumPy since
        # it's inherently sequential and tiny relative to the forward pass;
        # only the forward pass benefits from JIT.
        ids = np.asarray(input_ids, dtype=np.int64).copy()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.max_seq_len:])[:, -1, :] / max(temperature, 1e-5)
            if repetition_penalty and repetition_penalty > 1.0:
                for row, sequence in enumerate(ids):
                    seen = np.unique(sequence)
                    logits[row, seen] = np.where(
                        logits[row, seen] < 0,
                        logits[row, seen] * repetition_penalty,
                        logits[row, seen] / repetition_penalty,
                    )
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                excluded = np.argpartition(logits, -k, axis=1)[:, :-k]
                logits[np.arange(len(ids))[:, None], excluded] = -np.inf
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            if top_p is not None and 0 < top_p < 1:
                order = np.argsort(-probs, axis=1)
                sorted_probs = np.take_along_axis(probs, order, axis=1)
                cutoff = np.cumsum(sorted_probs, axis=1) > top_p
                cutoff[:, 0] = False
                probs[np.arange(len(ids))[:, None], order] = np.where(cutoff, 0.0, sorted_probs)
                probs /= probs.sum(1, keepdims=True)
            next_ids = np.array([np.random.choice(logits.shape[1], p=p) for p in probs])[:, None]
            ids = np.concatenate([ids, next_ids], axis=1)
            if eos_id is not None and np.all(next_ids == eos_id):
                break
        return ids


class JaxTabularMLP(_JaxFusedAdamMixin, TabularMLP):
    """The native tabular-MLP parameter layout with JAX math and gradients.

    Used automatically instead of the plain NumPy `TabularMLP` whenever the
    resolved device is `cuda`/`tpu`, so tabular training actually runs on
    the detected accelerator instead of silently staying on CPU.
    """

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop("precision", "fp32")
        super().__init__(*args, **kwargs)

    def _jax_params(self):
        _, jnp = _jax()
        dtype = {"fp16": jnp.float16, "bf16": jnp.bfloat16}.get(self.precision, jnp.float32)
        return {name: jnp.asarray(value.data, dtype=dtype) for name, value in self.named_parameters()}

    def _forward_jax(self, params, numeric, categorical, dropout_masks=None):
        jax, jnp = _jax()
        parts = [jnp.asarray(numeric, dtype=jnp.float32)] if self.n_numeric else []
        categorical = jnp.asarray(categorical, dtype=jnp.int32)
        for i in range(len(self.embeddings)):
            parts.append(params[f"embeddings.{i}"][categorical[:, i]])
        x = jnp.concatenate(parts, axis=1) if parts else jnp.asarray(numeric, dtype=jnp.float32)
        for i in range(len(self.weights)):
            preactivation = x @ params[f"weights.{i}"] + params[f"biases.{i}"]
            x = jax.nn.gelu(preactivation, approximate=True)
            if dropout_masks is not None and dropout_masks[i] is not None:
                x = x * dropout_masks[i] / (1.0 - self.dropout)
        out = x @ params["head_weight"] + params["head_bias"]
        if self.task == "regression":
            out = out[:, 0]
        return out

    def _make_dropout_masks(self, batch_size):
        if not self.training or self.dropout <= 0:
            return None
        _, jnp = _jax()
        return [
            jnp.asarray((np.random.random((batch_size, weight.data.shape[1])) >= self.dropout).astype(np.float32))
            for weight in self.weights
        ]

    def forward(self, numeric, categorical, cache=False):
        params = self._jax_params()
        batch_size = numeric.shape[0] if self.n_numeric else categorical.shape[0]
        dropout_masks = self._make_dropout_masks(batch_size)
        out = np.asarray(self._get_forward_jit()(params, numeric, categorical, dropout_masks))
        return (out, None) if cache else out

    def _get_forward_jit(self):
        if getattr(self, "_forward_jit_fn", None) is None:
            jax, _ = _jax()
            self._forward_jit_fn = jax.jit(self._forward_jax)
        return self._forward_jit_fn

    def _get_loss_and_grad_jit(self):
        if getattr(self, "_loss_and_grad_jit_fn", None) is None:
            jax, jnp = _jax()
            is_regression = self.task == "regression"

            def loss_fn(current, numeric, categorical, dropout_masks, target_arr):
                logits = self._forward_jax(current, numeric, categorical, dropout_masks)
                if is_regression:
                    error = logits - target_arr
                    return jnp.mean(error * error)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                return -jnp.mean(jnp.take_along_axis(log_probs, target_arr[:, None], axis=1))

            self._loss_and_grad_jit_fn = jax.jit(jax.value_and_grad(loss_fn))
        return self._loss_and_grad_jit_fn

    def loss_and_backward(self, numeric, categorical, target):
        jax, jnp = _jax()
        params = self._jax_params()
        batch_size = numeric.shape[0] if self.n_numeric else categorical.shape[0]
        dropout_masks = self._make_dropout_masks(batch_size)
        is_regression = self.task == "regression"
        target_arr = jnp.asarray(target, dtype=jnp.float32 if is_regression else jnp.int32)

        loss, gradients = self._get_loss_and_grad_jit()(params, numeric, categorical, dropout_masks, target_arr)
        for name, parameter in self.named_parameters():
            parameter.grad[...] = np.asarray(gradients[name], dtype=np.float32)
        return float(loss)

    def _get_fused_train_step(self, weight_decay: float, decoupled: bool):
        cache_key = (weight_decay, decoupled)
        if self._fused_train_step_key != cache_key:
            jax, jnp = _jax()
            is_regression = self.task == "regression"

            def loss_fn(params, numeric, categorical, dropout_masks, target_arr):
                logits = self._forward_jax(params, numeric, categorical, dropout_masks)
                if is_regression:
                    error = logits - target_arr
                    return jnp.mean(error * error)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                return -jnp.mean(jnp.take_along_axis(log_probs, target_arr[:, None], axis=1))

            grad_fn = jax.value_and_grad(loss_fn)

            def step(params, m, v, step_count, numeric, categorical, dropout_masks, target_arr, lr, grad_clip):
                loss, grads = grad_fn(params, numeric, categorical, dropout_masks, target_arr)
                grad_norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grads.values()))
                clip_scale = jnp.where(grad_clip > 0, jnp.minimum(1.0, grad_clip / (grad_norm + 1e-12)), 1.0)
                grads = {name: g * clip_scale for name, g in grads.items()}
                new_params, new_m, new_v = _adam_update_tree(
                    jnp, params, m, v, grads, step_count, lr, weight_decay, decoupled
                )
                return new_params, new_m, new_v, loss, grad_norm

            self._fused_train_step_fn = jax.jit(step)
            self._fused_train_step_key = cache_key
        return self._fused_train_step_fn

    def fused_train_step(self, numeric, categorical, target, lr, weight_decay, grad_clip, decoupled):
        """One fused forward+backward+AdamW step, entirely on-device.

        Call `init_fused_state()` once before the first step and
        `sync_fused_to_host()` before reading `state_dict()`/`forward()` on
        the host (e.g. for validation or checkpointing).
        """
        if self._fused_params is None:
            raise RuntimeError("init_fused_state() must be called before fused_train_step().")
        jax, jnp = _jax()
        step_fn = self._get_fused_train_step(weight_decay, decoupled)
        self._fused_step += 1
        batch_size = numeric.shape[0] if self.n_numeric else categorical.shape[0]
        dropout_masks = self._make_dropout_masks(batch_size)
        is_regression = self.task == "regression"
        target_arr = jnp.asarray(target, dtype=jnp.float32 if is_regression else jnp.int32)
        new_params, new_m, new_v, loss, grad_norm = step_fn(
            self._fused_params, self._fused_m, self._fused_v,
            jnp.asarray(self._fused_step, dtype=jnp.float32),
            numeric, categorical, dropout_masks, target_arr,
            jnp.asarray(lr, dtype=jnp.float32), jnp.asarray(grad_clip, dtype=jnp.float32),
        )
        self._fused_params, self._fused_m, self._fused_v = new_params, new_m, new_v
        return float(loss), float(grad_norm)
