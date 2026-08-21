"""Automatic configuration.

Turns a `Dataset` + a user-supplied `TrainConfig` (with mostly-None fields)
into a fully-resolved `ResolvedConfig`. Every automatic choice is driven by
simple, explainable heuristics based on dataset size -- this is not meant
to be state-of-the-art NAS, it's meant to produce a *sane, working* default
so `tl.train("./data")` just works.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import TrainConfig, ResolvedConfig
from ..data.loader import Dataset
from ..devices.device import auto_select_device
from .detector import detect_task, target_column


def _auto_model_size(n_examples: int, kind: str):
    """Return (d_model, layers, heads, ff_mult) scaled to dataset size.

    Small datasets get small models (avoids absurd overfitting / long CPU
    training times); larger datasets get bigger models. These are
    deliberately modest sizes suitable for CPU-friendly experimentation --
    users can always override with layers=, d_model=, etc.
    """
    if kind in ("text", "text_labeled"):
        if n_examples < 200:
            return 64, 2, 2, 2
        elif n_examples < 5000:
            return 128, 4, 4, 2
        elif n_examples < 50000:
            return 256, 6, 8, 4
        else:
            return 384, 8, 8, 4
    else:  # tabular
        if n_examples < 500:
            return 32, 2, 1, 2
        elif n_examples < 20000:
            return 64, 3, 1, 2
        else:
            return 128, 4, 1, 2


def _auto_batch_size(n_examples: int) -> int:
    if n_examples < 200:
        return 8
    elif n_examples < 2000:
        return 16
    elif n_examples < 20000:
        return 32
    else:
        return 64


def _auto_epochs(n_examples: int) -> int:
    if n_examples < 200:
        return 40
    elif n_examples < 2000:
        return 20
    elif n_examples < 20000:
        return 10
    else:
        return 5


def resolve_config(ds: Dataset, user: TrainConfig) -> ResolvedConfig:
    n = len(ds)
    task = user.task or detect_task(ds)
    model_type = user.model_type or (
        "transformer" if task in ("text-generation", "text-classification") else "mlp"
    )
    tokenizer = user.tokenizer or "bpe"
    if tokenizer not in ("char", "bpe"):
        raise ValueError("tokenizer must be 'char' or 'bpe'")

    d_model, layers, heads, ff_mult = _auto_model_size(n, ds.kind)

    device, precision = auto_select_device(user.device, user.precision)

    out = user.out or "model.tl"
    checkpoint_dir = user.checkpoint_dir or (out + ".ckpt")

    resolved = ResolvedConfig(
        out=out,
        force=bool(user.force),
        resume=user.resume,
        ask_on_data_change=bool(user.ask_on_data_change),
        task=task,
        model_type=model_type,
        d_model=user.d_model or d_model,
        layers=user.layers or layers,
        heads=user.heads or heads,
        ff_mult=user.ff_mult or ff_mult,
        dropout=user.dropout if user.dropout is not None else 0.1,
        max_seq_len=user.max_seq_len or (256 if ds.kind in ("text", "text_labeled") else 1),
        tokenizer=tokenizer,
        bpe_vocab_size=user.bpe_vocab_size or 1000,
        optimizer=user.optimizer or "adamw",
        learning_rate=user.learning_rate or (3e-4 if model_type == "transformer" else 1e-3),
        weight_decay=user.weight_decay if user.weight_decay is not None else 0.01,
        batch_size=user.batch_size or _auto_batch_size(n),
        epochs=user.epochs or _auto_epochs(n),
        max_steps=user.max_steps,
        grad_clip=user.grad_clip if user.grad_clip is not None else 1.0,
        warmup_steps=user.warmup_steps if user.warmup_steps is not None else min(100, max(1, n // 10)),
        val_split=user.val_split if user.val_split is not None else (0.1 if n >= 50 else 0.0),
        patience=user.patience or 5,
        min_delta=user.min_delta if user.min_delta is not None else 1e-4,
        device=device,
        precision=precision,
        checkpoint_every=user.checkpoint_every or 50,
        checkpoint_dir=checkpoint_dir,
        seed=user.seed,
        verbose=user.verbose,
    )
    return resolved
