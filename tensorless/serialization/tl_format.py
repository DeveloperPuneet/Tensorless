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

import pickle

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


def _migrate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize supported legacy payloads without changing model weights."""
    migrated = dict(payload)
    migrated.setdefault("tensorless_version", "unknown")
    migrated.setdefault("metrics", {})
    migrated.setdefault("training_complete", True)
    migrated.setdefault("dataset_fingerprint", None)
    migrated.setdefault("tokenizer_state", None)
    migrated.setdefault("preprocessor_state", None)

    config = migrated.get("config")
    if isinstance(config, dict):
        config.setdefault("tokenizer", "char")
        config.setdefault("bpe_vocab_size", 1000)
        config.setdefault("precision", "fp32")
        config.setdefault("device", "cpu")
        migrated["config"] = config

    tokenizer_state = migrated.get("tokenizer_state")
    if isinstance(tokenizer_state, dict):
        tokenizer_state.setdefault("tokenizer_type", "char")
        migrated["tokenizer_state"] = tokenizer_state

    return migrated


def save_tl(path: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("tl_format_version", TL_FORMAT_VERSION)
    payload.setdefault("tensorless_version", __version__)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SerializationError(f"Failed to write .tl file to '{path}': {e}") from e


def load_tl(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise SerializationError(f"'{path}' does not exist.")
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as e:
        raise SerializationError(f"Failed to read .tl file '{path}': {e}") from e

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SerializationError(
            f"'{path}' is missing required field(s) {missing}. It may be "
            f"corrupt or not a valid Tensorless .tl file."
        )

    file_version = payload.get("tl_format_version")
    if not isinstance(file_version, int):
        raise SerializationError(
            f"'{path}' has an invalid .tl format version. It may be corrupt."
        )
    if file_version > TL_FORMAT_VERSION:
        raise SerializationError(
            f"'{path}' was created with a newer .tl format (v{file_version}) "
            f"than this installed version of Tensorless supports "
            f"(v{TL_FORMAT_VERSION}). Please upgrade Tensorless."
        )

    return _migrate_payload(payload)
