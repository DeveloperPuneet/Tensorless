# Checkpointing & Resume

## Where checkpoints live

Every training run writes to a checkpoint directory, by default
`<out>.ckpt/` (e.g. `model.tl.ckpt/checkpoint.pt`). You never need to
create or manage this directory yourself.

## What's in a checkpoint

`tensorless/checkpoint/manager.py` writes a single file,
`checkpoint.pt`, containing:

- `model_state_dict` — model weights
- `best_model_state_dict` — best validation weights, when validation is enabled
- `optimizer_state_dict` — optimizer momentum/variance buffers
- `scheduler_state_dict` — learning rate schedule position
- `epoch`, `global_step` — where training left off
- `early_stopping_best`, `early_stopping_bad_checks` — early stopping state
- `config` — the fully resolved training configuration used
- `meta` — task-specific sizing info (vocab size, number of classes, etc.)
- `tokenizer_state` / `preprocessor_state` — whichever applies to the task
- `dataset_fingerprint` — the fingerprint of the dataset this checkpoint
  was trained on (see [automatic_mode.md](automatic_mode.md#5-the-smart-auto-check))
- `training_complete` — whether this checkpoint represents a finished run

This is everything needed to either resume training or reconstruct the
final `.tl` file — nothing about resumption depends on the original
dataset still being on disk in the same location, only on it being
fingerprint-identical to what was originally used.

Set `gradient_checkpointing=True` to reduce native transformer activation
memory. Backward recomputes each block and replays its dropout state, trading
extra compute for lower peak memory. The JAX backend uses `jax.checkpoint`
for the same option and uses device collectives for local multi-device
gradient reduction.

## When checkpoints are written

- Every `checkpoint_every` steps (default: 50) during training, with
  `training_complete=False`
- At the end of every epoch, with `training_complete` set to whether
  that was the last epoch
- On early stopping, with `training_complete=True`

Writes are atomic: Tensorless writes to a temporary file in the same
directory and renames it into place, so a crash mid-write never leaves a
corrupt checkpoint that would block resumption.

JAX and MLX accelerator arrays are converted to portable NumPy arrays before
they enter a `.tl` file or checkpoint. This keeps models loadable on a CPU
machine. Set `checkpoint_shard_size_mb` above zero to store model and
optimizer state in independently readable shard files plus a small
`checkpoint.pt` manifest. The normal resume API reconstructs the state
automatically.

When loading `.tl` files, Tensorless fills compatible fields introduced by
older versions with safe defaults. Files created by a newer unsupported format
version are rejected with an upgrade message instead of being partially read.

## How resumption works

When `tl.train()` finds a checkpoint whose `dataset_fingerprint` matches
the current dataset and `training_complete=False`, it:

1. Loads the tokenizer/preprocessor state from the checkpoint (so the
   vocabulary or column encoding is identical to the original run)
2. Rebuilds the exact same model architecture from the checkpoint's
   saved `config`
3. Loads model, optimizer, and scheduler state
4. Continues the training loop from the saved `epoch` / `global_step`

**Important:** any config overrides you pass to the resuming
`tl.train()` call (e.g. a different `d_model`) are ignored in favor of
the checkpoint's original config. This is intentional — a resumed run
must use the same architecture as the interrupted run, or the saved
weights simply won't fit. If you want different hyperparameters,
either delete the checkpoint directory first or pass `force=True`.

## Manually clearing a checkpoint

```python
from tensorless.checkpoint.manager import CheckpointManager
CheckpointManager("model.tl.ckpt").clear()
```

or simply delete the directory:

```bash
rm -rf model.tl.ckpt
```

`tl.train(..., force=True)` does this for you automatically before
retraining.
