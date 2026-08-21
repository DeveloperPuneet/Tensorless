"""MLP model for tabular data (classification / regression).

Numeric columns feed directly into the network; each categorical column
gets its own small embedding table, and all features are concatenated
before the hidden layers.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _embedding_dim(vocab_size: int) -> int:
    # Common rule of thumb, capped to keep tiny-data models tiny.
    return max(2, min(32, round(1.6 * (vocab_size ** 0.56))))


class TabularMLP(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        categorical_vocab_sizes: List[int],
        d_model: int,
        layers: int,
        dropout: float,
        task: str,
        n_classes: int = 0,
    ):
        super().__init__()
        self.task = task
        self.n_numeric = n_numeric
        self.categorical_vocab_sizes = categorical_vocab_sizes

        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, _embedding_dim(v)) for v in categorical_vocab_sizes]
        )
        cat_dim = sum(_embedding_dim(v) for v in categorical_vocab_sizes)
        in_dim = n_numeric + cat_dim

        dims = [in_dim] + [d_model] * layers
        modules = []
        for i in range(len(dims) - 1):
            modules += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*modules)

        out_dim = n_classes if task == "classification" else 1
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_numeric > 0:
            parts.append(numeric)
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, i]))
        x = torch.cat(parts, dim=1) if parts else numeric
        x = self.backbone(x)
        out = self.head(x)
        if self.task == "regression":
            return out.squeeze(-1)
        return out
