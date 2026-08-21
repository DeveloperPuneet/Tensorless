"""The `.tl` file format.

A `.tl` file is a single portable file (a torch pickle archive under the
hood) containing everything needed for inference on a *different*
machine, with no access to the original dataset or training code:

    {
        "tl_format_version": int,
        "tensorless_version": str,
        "task": str,                  # e.g. "text-generation"
        "model_type": str,            # e.g. "transformer"
        "config": {...},              # resolved training config
        "meta": {...},                # vocab_size / n_classes / column info
        "model_state_dict": {...},
        "tokenizer_state": {...} | None,
        "preprocessor_state": {...} | None,
        "dataset_fingerprint": str,
        "training_complete": bool,
        "metrics": {...},
    }

We deliberately use a single file (rather than a directory/zip of many
files) so users can `scp`/email/upload one `model.tl` and have it just
work elsewhere, per the framework's "portable single file" requirement.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from .._version import __version__, TL_FORMAT_VERSION
from ..errors import SerializationError

REQUIRED_KEYS = (
    "tl_format_version",
    "task",
    "model_type",
    "config",
    "meta",
    "model_state_dict",
)


def save_tl(path: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("tl_format_version", TL_FORMAT_VERSION)
    payload.setdefault("tensorless_version", __version__)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = path + ".tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SerializationError(f"Failed to write .tl file to '{path}': {e}") from e


def load_tl(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise SerializationError(f"'{path}' does not exist.")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise SerializationError(f"Failed to read .tl file '{path}': {e}") from e

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SerializationError(
            f"'{path}' is missing required field(s) {missing}. It may be "
            f"corrupt or not a valid Tensorless .tl file."
        )

    file_version = payload.get("tl_format_version")
    if file_version > TL_FORMAT_VERSION:
        raise SerializationError(
            f"'{path}' was created with a newer .tl format (v{file_version}) "
            f"than this installed version of Tensorless supports "
            f"(v{TL_FORMAT_VERSION}). Please upgrade Tensorless."
        )

    return payload
