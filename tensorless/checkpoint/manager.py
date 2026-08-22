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
MANIFEST_KEY = "tensorless_sharded_checkpoint"


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
        shard_size = int(state.get("config", {}).get("checkpoint_shard_size_mb", 0) or 0) * 1024 * 1024
        if shard_size > 0:
            self._save_sharded(state, shard_size)
            return
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

    def _save_sharded(self, state: Dict[str, Any], shard_size: int) -> None:
        try:
            for name in os.listdir(self.checkpoint_dir):
                if name.startswith("checkpoint.shard-"):
                    os.remove(os.path.join(self.checkpoint_dir, name))
            manifest = {key: value for key, value in state.items()
                        if key not in ("model_state_dict", "optimizer_state_dict")}
            manifest[MANIFEST_KEY] = True
            manifest["model_shards"] = self._write_shards("model", state["model_state_dict"], shard_size)
            manifest["optimizer_shards"] = self._write_shards("optimizer", state["optimizer_state_dict"], shard_size)
            fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
            os.close(fd)
            with open(tmp_path, "wb") as handle:
                pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(tmp_path, self.path)
        except Exception as e:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise CheckpointError(f"Failed to save sharded checkpoint to '{self.path}': {e}") from e

    def _write_shards(self, kind: str, values: Dict[str, Any], shard_size: int):
        shards = []
        current = {}
        current_size = 0
        for key, value in values.items():
            value_size = len(pickle.dumps({key: value}, protocol=pickle.HIGHEST_PROTOCOL))
            if current and current_size + value_size > shard_size:
                shards.append(self._write_shard(kind, len(shards), current))
                current, current_size = {}, 0
            current[key] = value
            current_size += value_size
        if current or not shards:
            shards.append(self._write_shard(kind, len(shards), current))
        return shards

    def _write_shard(self, kind: str, index: int, values: Dict[str, Any]) -> str:
        filename = f"checkpoint.shard-{kind}-{index:04d}.pkl"
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        return filename

    def load(self, map_location: Optional[str] = None) -> Dict[str, Any]:
        if not self.exists():
            raise CheckpointError(f"No checkpoint found at '{self.path}'.")
        try:
            with open(self.path, "rb") as handle:
                state = pickle.load(handle)
            if state.get(MANIFEST_KEY):
                state["model_state_dict"] = self._load_shards(state.pop("model_shards"))
                state["optimizer_state_dict"] = self._load_shards(state.pop("optimizer_shards"))
            return state
        except Exception as e:
            raise CheckpointError(
                f"Checkpoint at '{self.path}' is corrupt or incompatible: {e}"
            ) from e

    def _load_shards(self, filenames):
        values = {}
        for filename in filenames:
            path = os.path.join(self.checkpoint_dir, filename)
            with open(path, "rb") as handle:
                values.update(pickle.load(handle))
        return values

    def clear(self) -> None:
        if os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
