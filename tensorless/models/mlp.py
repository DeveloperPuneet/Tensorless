"""NumPy MLP for tabular classification and regression."""

from __future__ import annotations

from typing import List
import numpy as np

from ..engine import Module, Parameter


def _embedding_dim(vocab_size: int) -> int:
    return max(2, min(32, round(1.6 * (vocab_size ** 0.56))))


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


class TabularMLP(Module):
    def __init__(self, n_numeric, categorical_vocab_sizes: List[int], d_model, layers, dropout, task, n_classes=0):
        self.training = True
        self.task = task
        self.dropout = dropout
        self.n_numeric = n_numeric
        self.categorical_vocab_sizes = categorical_vocab_sizes
        self.embeddings = [Parameter(np.random.normal(0, .02, (v, _embedding_dim(v))).astype(np.float32)) for v in categorical_vocab_sizes]
        in_dim = n_numeric + sum(_embedding_dim(v) for v in categorical_vocab_sizes)
        self.weights, self.biases = [], []
        for i in range(layers):
            fan_in = in_dim if i == 0 else d_model
            self.weights.append(Parameter(np.random.normal(0, .02, (fan_in, d_model)).astype(np.float32)))
            self.biases.append(Parameter(np.zeros(d_model, dtype=np.float32)))
        out_dim = n_classes if task == "classification" else 1
        self.head_weight = Parameter(np.random.normal(0, .02, (d_model if layers else in_dim, out_dim)).astype(np.float32))
        self.head_bias = Parameter(np.zeros(out_dim, dtype=np.float32))

    def forward(self, numeric, categorical, cache=False):
        parts = [numeric.astype(np.float32)] if self.n_numeric else []
        for i, embedding in enumerate(self.embeddings):
            parts.append(embedding.data[categorical[:, i]])
        x = np.concatenate(parts, axis=1) if parts else numeric.astype(np.float32)
        activations = [x]
        preactivations = []
        dropout_masks = []
        for weight, bias in zip(self.weights, self.biases):
            preactivation = x @ weight.data + bias.data
            x = _gelu(preactivation)
            x, dropout_mask = _dropout(x, self.dropout, self.training)
            preactivations.append(preactivation)
            activations.append(x)
            dropout_masks.append(dropout_mask)
        out = x @ self.head_weight.data + self.head_bias.data
        if self.task == "regression": out = out[:, 0]
        return (out, (activations, preactivations, categorical, dropout_masks)) if cache else out

    def loss_and_backward(self, numeric, categorical, target):
        logits, (activations, preactivations, categorical, dropout_masks) = self.forward(numeric, categorical, cache=True)
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
            if dropout_masks[i] is not None:
                dx *= dropout_masks[i] / (1.0 - self.dropout)
            dx *= _gelu_backward(preactivations[i])
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
