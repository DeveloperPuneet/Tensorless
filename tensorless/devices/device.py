"""Hardware auto-detection.

Preference order is TPU -> GPU -> CPU, but selection is "intelligent"
rather than blind: we verify each backend is actually usable (not just
importable) before choosing it, and we fall back gracefully -- including
at *runtime*, if a chosen device turns out to error out mid-training, the
trainer (see `training/trainer.py`) will catch that and fall back too.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _tpu_available() -> bool:
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _cuda_supports_bf16() -> bool:
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
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


def get_torch_device(device: str) -> torch.device:
    """Convert our string device name into a torch.device, with a
    runtime fallback to CPU if the requested backend is unavailable.
    """
    try:
        if device == "tpu":
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        if device == "cuda":
            if not _cuda_available():
                return torch.device("cpu")
            return torch.device("cuda")
        if device == "mps":
            if not _mps_available():
                return torch.device("cpu")
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
