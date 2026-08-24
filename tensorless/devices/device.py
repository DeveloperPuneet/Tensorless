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


_BF16_CAPABLE_GPU_HINTS = (
    "a100", "a10", "a30", "a40", "l4", "l40", "l20",
    "h100", "h200", "h800", "b100", "b200", "gh200",
    "rtx 30", "rtx 40", "rtx 50", "rtx a", "geforce rtx 30", "geforce rtx 40", "geforce rtx 50",
)


def _cuda_supports_bf16() -> bool:
    """Best-effort detection of real (fast) bf16 tensor-core support.

    Older GPUs (Turing/Volta/Pascal -- T4, V100, P100, K80, RTX 20xx) either
    lack bf16 tensor cores or emulate bf16 slowly, so fp16 (with loss
    scaling) is the safer/faster default there. Ampere-or-newer GPUs (A100,
    L4, H100, RTX 30xx+, ...) run bf16 at full tensor-core speed and don't
    need loss scaling, so we opt in automatically when the detected GPU name
    matches a known Ampere-or-newer architecture. Anything unrecognized
    conservatively falls back to fp16, matching the previous behavior.
    """
    try:
        import jax
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            return False
        name = getattr(gpus[0], "device_kind", "").lower()
        return any(hint in name for hint in _BF16_CAPABLE_GPU_HINTS)
    except (ImportError, RuntimeError):
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


