"""Model registry.

Central place that knows how to build a model given a task + resolved
config + the dataset-derived metadata (vocab size, n_classes, etc). Kept
separate from `auto/config.py` so new model types/backends can be added
here without touching auto-configuration logic, per the "extensible
architecture" requirement.
"""

from __future__ import annotations

from typing import Any, Dict

from .transformer import TinyTransformer
from .mlp import TabularMLP
from ..errors import ModelError


def build_model(task: str, model_type: str, cfg: Dict[str, Any], meta: Dict[str, Any]):
    """Build a fresh, randomly-initialized model.

    `cfg` is the resolved training config (dict). `meta` carries
    task-specific sizing info produced during data prep, e.g.:
      - text tasks: {"vocab_size": int, "pad_id": int, "n_classes": int}
      - tabular tasks: {"n_numeric": int, "categorical_vocab_sizes": [...], "n_classes": int}
    """
    if model_type == "transformer":
        return TinyTransformer(
            vocab_size=meta["vocab_size"],
            d_model=cfg["d_model"],
            layers=cfg["layers"],
            heads=cfg["heads"],
            ff_mult=cfg["ff_mult"],
            dropout=cfg["dropout"],
            max_seq_len=cfg["max_seq_len"],
            task=task,
            n_classes=meta.get("n_classes", 0),
            pad_id=meta.get("pad_id", 0),
        )
    elif model_type == "mlp":
        return TabularMLP(
            n_numeric=meta["n_numeric"],
            categorical_vocab_sizes=meta["categorical_vocab_sizes"],
            d_model=cfg["d_model"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            task=task,
            n_classes=meta.get("n_classes", 0),
        )
    else:
        raise ModelError(f"Unknown model_type '{model_type}'.")
