"""Hardware auto-detection.

Preference order is TPU -> GPU -> CPU, but selection is "intelligent"
rather than blind: we verify each backend is actually usable (not just
importable) before choosing it, and we fall back gracefully -- including
at *runtime*, if a chosen device turns out to error out mid-training, the
trainer (see `training/trainer.py`) will catch that and fall back too.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _tpu_available() -> bool:
    try:
        import jax
        return any(device.platform == "tpu" for device in jax.devices())
    except (ImportError, RuntimeError):
        return False


def _cuda_available() -> bool:
    try:
        import jax
        return any(device.platform == "gpu" for device in jax.devices())
    except (ImportError, RuntimeError):
        return False


def _mps_available() -> bool:
    try:
        import mlx.core as mx
        mx.array(0).item()
        return True
    except (ImportError, RuntimeError):
        return False


def _cuda_supports_bf16() -> bool:
    """Use the conservative CUDA default unless hardware is identified."""
    return False


def auto_select_device(user_device: Optional[str], user_precision: Optional[str]) -> Tuple[str, str]:
    """Resolve the device and precision to use.

    `user_device` / `user_precision` are honored if given (with a
    graceful downgrade if the requested device isn't actually available).
    Otherwise we pick automatically: tpu > cuda > mps > cpu.
    """
    if user_device is not None:
        device = user_device
        if device == "tpu" and not _tpu_available():
            device = "cuda" if _cuda_available() else "cpu"
        elif device == "cuda" and not _cuda_available():
            device = "cpu"
        elif device == "mps" and not _mps_available():
            device = "cpu"
    else:
        if _tpu_available():
            device = "tpu"
        elif _cuda_available():
            device = "cuda"
        elif _mps_available():
            device = "mps"
        else:
            device = "cpu"

    if user_precision is not None:
        precision = user_precision
    else:
        if device == "cuda" and _cuda_supports_bf16():
            precision = "bf16"
        elif device == "cuda":
            precision = "fp16"
        elif device == "tpu":
            precision = "bf16"
        else:
            # CPU and MPS: stick to fp32 for correctness/stability by default.
            precision = "fp32"

    return device, precision


def get_device(device: str) -> str:
    """Return the selected usable device for the native backend."""
    if device == "mps" and _mps_available():
        return "mps"
    if device in ("cuda", "tpu"):
        try:
            import jax
            platform = "gpu" if device == "cuda" else device
            if any(d.platform == platform for d in jax.devices()):
                return device
        except (ImportError, RuntimeError):
            pass
    return "cpu"


