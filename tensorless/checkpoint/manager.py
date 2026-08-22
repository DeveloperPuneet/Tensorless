"""Checkpoint management.

Handles all the state needed to resume training safely and transparently:
model weights, optimizer state, scheduler state, epoch/step counters, the
resolved training config, tokenizer/preprocessor state, the dataset
fingerprint used for training, and the best-metric-so-far for early
stopping.

Users never touch this directly -- `tl.train()` decides automatically
whether to create, update, or resume from a checkpoint (see
`training/trainer.py` and the "Smart Auto Check" logic in `api.py`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import pickle

from ..errors import CheckpointError

CHECKPOINT_FILENAME = "checkpoint.pt"


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically write `state` to the checkpoint file.

        Writes to a temp file first and renames it into place, so a crash
        or interruption mid-write never leaves a corrupt checkpoint that
        would block resumption.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(tmp_path, self.path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise CheckpointError(f"Failed to save checkpoint to '{self.path}': {e}") from e

    def load(self, map_location: Optional[str] = None) -> Dict[str, Any]:
        if not self.exists():
            raise CheckpointError(f"No checkpoint found at '{self.path}'.")
        try:
            with open(self.path, "rb") as handle:
                return pickle.load(handle)
        except Exception as e:
            raise CheckpointError(
                f"Checkpoint at '{self.path}' is corrupt or incompatible: {e}"
            ) from e

    def clear(self) -> None:
        if os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
