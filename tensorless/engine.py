"""Small NumPy training engine used by Tensorless models."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator
import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)

    def numel(self) -> int:
        return int(self.data.size)


class Module:
    def parameters(self) -> Iterator[Parameter]:
        seen = set()

        def visit(value):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from visit(item)

        for value in self.__dict__.values():
            yield from visit(value)

    def named_parameters(self, prefix: str = ""):
        yield from self._named_parameters(prefix, set())

    def _named_parameters(self, prefix: str, seen):
        for name, value in self.__dict__.items():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield full, value
            elif isinstance(value, Module):
                yield from value._named_parameters(full, seen)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Parameter):
                        if id(item) not in seen:
                            seen.add(id(item))
                            yield f"{full}.{i}", item
                    elif isinstance(item, Module):
                        yield from item._named_parameters(f"{full}.{i}", seen)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        current = dict(self.named_parameters())
        missing = [name for name in current if name not in state]
        if missing:
            raise ValueError(f"Missing model parameters: {missing}")
        for name, parameter in current.items():
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != parameter.data.shape:
                raise ValueError(f"Shape mismatch for {name}: {value.shape} != {parameter.data.shape}")
            parameter.data[...] = value

    def train(self, mode: bool = True):
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)


def layer_norm_forward(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
    """Normalize over the last axis, then apply a learned scale/shift.

    Returns the normalized output plus a cache used by `layer_norm_backward`.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = x_hat * gain + bias
    return out.astype(np.float32), (x_hat, inv_std, gain)


def layer_norm_backward(grad_out: np.ndarray, cache):
    """Backward pass matching `layer_norm_forward`.

    Returns (grad_x, grad_gain, grad_bias).
    """
    x_hat, inv_std, gain = cache
    d = grad_out.shape[-1]
    flat_grad_out = grad_out.reshape(-1, d)
    flat_x_hat = x_hat.reshape(-1, d)
    grad_gain = (flat_grad_out * flat_x_hat).sum(axis=0)
    grad_bias = flat_grad_out.sum(axis=0)
    grad_x_hat = grad_out * gain
    grad_x = inv_std / d * (
        d * grad_x_hat
        - grad_x_hat.sum(axis=-1, keepdims=True)
        - x_hat * (grad_x_hat * x_hat).sum(axis=-1, keepdims=True)
    )
    return grad_x.astype(np.float32), grad_gain.astype(np.float32), grad_bias.astype(np.float32)


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray, ignore_index=None):
    flat = logits.reshape(-1, logits.shape[-1]).astype(np.float64)
    target = targets.reshape(-1).astype(np.int64)
    valid = np.ones(len(target), dtype=bool) if ignore_index is None else target != ignore_index
    if not np.any(valid):
        return 0.0, np.zeros_like(logits)
    shifted = flat - flat.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    rows = np.arange(len(target))[valid]
    loss = -np.log(np.maximum(probs[rows, target[valid]], 1e-12)).mean()
    grad = np.zeros_like(flat, dtype=np.float32)
    grad[valid] = probs[valid].astype(np.float32)
    grad[rows, target[valid]] -= 1.0
    grad[valid] /= valid.sum()
    return float(loss), grad.reshape(logits.shape)


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter], lr: float, weight_decay: float = 0.0):
        self.parameters_list = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0

    def zero_grad(self):
        for parameter in self.parameters_list:
            parameter.grad.fill(0.0)

    def state_dict(self):
        return {"step": self.step_count, "lr": self.lr, "weight_decay": self.weight_decay}

    def load_state_dict(self, state):
        self.step_count = int(state.get("step", 0))
        self.lr = float(state.get("lr", self.lr))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))


class SGD(Optimizer):
    def __init__(self, parameters, lr, weight_decay=0.0, momentum=0.9):
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters_list]

    def step(self):
        self.step_count += 1
        for p, velocity in zip(self.parameters_list, self.velocity):
            velocity *= self.momentum
            velocity += p.grad + self.weight_decay * p.data
            p.data -= self.lr * velocity

    def state_dict(self):
        state = super().state_dict()
        state["velocity"] = [v.copy() for v in self.velocity]
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for target, source in zip(self.velocity, state.get("velocity", [])):
            target[...] = source


class Adam(Optimizer):
    def __init__(self, parameters, lr, weight_decay=0.0, decoupled=False):
        super().__init__(parameters, lr, weight_decay)
        self.decoupled = decoupled
        self.m = [np.zeros_like(p.data) for p in self.parameters_list]
        self.v = [np.zeros_like(p.data) for p in self.parameters_list]

    def step(self):
        self.step_count += 1
        for p, m, v in zip(self.parameters_list, self.m, self.v):
            if self.decoupled:
                p.data *= 1.0 - self.lr * self.weight_decay
                gradient = p.grad
            else:
                gradient = p.grad + self.weight_decay * p.data
            m *= 0.9
            m += 0.1 * gradient
            v *= 0.999
            v += 0.001 * gradient * gradient
            m_hat = m / (1.0 - 0.9 ** self.step_count)
            v_hat = v / (1.0 - 0.999 ** self.step_count)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def state_dict(self):
        state = super().state_dict()
        state.update({"m": [x.copy() for x in self.m], "v": [x.copy() for x in self.v]})
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for target, source in zip(self.m, state.get("m", [])):
            target[...] = source
        for target, source in zip(self.v, state.get("v", [])):
            target[...] = source


class LambdaScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lr = optimizer.lr
        self.optimizer.lr = self.base_lr * self._factor(0)

    def _factor(self, step):
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return max(
            0.1,
            1.0 - (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps),
        )

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.base_lr * self._factor(self.step_count)

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = int(state.get("step_count", 0))
        self.optimizer.lr = self.base_lr * self._factor(self.step_count)