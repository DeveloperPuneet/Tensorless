"""NumPy MLP for tabular classification and regression."""

from __future__ import annotations

from typing import List
import numpy as np

from ..engine import Module, Parameter


def _embedding_dim(vocab_size: int) -> int:
    return max(2, min(32, round(1.6 * (vocab_size ** 0.56))))


class TabularMLP(Module):
    def __init__(self, n_numeric, categorical_vocab_sizes: List[int], d_model, layers, dropout, task, n_classes=0):
        self.training = True
        self.task = task
        self.n_numeric = n_numeric
        self.categorical_vocab_sizes = categorical_vocab_sizes
        rng = np.random.default_rng()
        self.embeddings = [Parameter(rng.normal(0, .02, (v, _embedding_dim(v))).astype(np.float32)) for v in categorical_vocab_sizes]
        in_dim = n_numeric + sum(_embedding_dim(v) for v in categorical_vocab_sizes)
        self.weights, self.biases = [], []
        for i in range(layers):
            fan_in = in_dim if i == 0 else d_model
            self.weights.append(Parameter(rng.normal(0, .02, (fan_in, d_model)).astype(np.float32)))
            self.biases.append(Parameter(np.zeros(d_model, dtype=np.float32)))
        out_dim = n_classes if task == "classification" else 1
        self.head_weight = Parameter(rng.normal(0, .02, (d_model if layers else in_dim, out_dim)).astype(np.float32))
        self.head_bias = Parameter(np.zeros(out_dim, dtype=np.float32))

    def forward(self, numeric, categorical, cache=False):
        parts = [numeric.astype(np.float32)] if self.n_numeric else []
        for i, embedding in enumerate(self.embeddings):
            parts.append(embedding.data[categorical[:, i]])
        x = np.concatenate(parts, axis=1) if parts else numeric.astype(np.float32)
        activations = [x]
        for weight, bias in zip(self.weights, self.biases):
            x = np.maximum(0, x @ weight.data + bias.data)
            activations.append(x)
        out = x @ self.head_weight.data + self.head_bias.data
        if self.task == "regression": out = out[:, 0]
        return (out, (activations, categorical)) if cache else out

    def loss_and_backward(self, numeric, categorical, target):
        logits, (activations, categorical) = self.forward(numeric, categorical, cache=True)
        if self.task == "regression":
            error = logits - target
            loss = float(np.mean(error * error))
            grad = (2.0 / len(target)) * error[:, None]
        else:
            shifted = logits - logits.max(1, keepdims=True)
            probs = np.exp(shifted); probs /= probs.sum(1, keepdims=True)
            loss = float(-np.log(np.maximum(probs[np.arange(len(target)), target], 1e-12)).mean())
            grad = probs; grad[np.arange(len(target)), target] -= 1; grad /= len(target)
        self.head_weight.grad[...] = activations[-1].T @ grad
        self.head_bias.grad[...] = grad.sum(0)
        dx = grad @ self.head_weight.data.T
        for i in range(len(self.weights) - 1, -1, -1):
            dx = dx * (activations[i + 1] > 0)
            self.weights[i].grad[...] = activations[i].T @ dx
            self.biases[i].grad[...] = dx.sum(0)
            dx = dx @ self.weights[i].data.T
        offset = self.n_numeric
        for i, embedding in enumerate(self.embeddings):
            width = embedding.data.shape[1]
            embedding.grad.fill(0)
            np.add.at(embedding.grad, categorical[:, i], dx[:, offset:offset + width])
            offset += width
        return loss
