"""Backend-neutral accelerator memory controls."""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional


def clear_memory(device: Optional[str] = None) -> None:
    """Release Python and available backend caches without importing PyTorch."""
    gc.collect()
    if device in (None, "cuda", "tpu"):
        try:
            import jax
            jax.clear_caches()
        except ImportError:
            pass
    if device in (None, "mps"):
        try:
            import mlx.core as mx
            mx.eval()
        except (ImportError, RuntimeError):
            pass


def memory_stats(device: Optional[str] = None) -> Dict[str, Any]:
    """Return available backend memory statistics, or an empty dictionary."""
    if device in (None, "cuda", "tpu"):
        try:
            import jax
            devices = jax.devices()
            stats = devices[0].memory_stats() if devices else None
            return dict(stats or {})
        except (ImportError, RuntimeError, AttributeError):
            pass
    return {}
