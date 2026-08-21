# Automatic Mode

Tensorless's whole premise is that `tl.train("./data")` should just
work. This page explains exactly what "automatic" means at each stage,
so the behavior is predictable rather than magic.

## 1. Task detection

`tensorless/auto/detector.py` looks at the *shape* of your loaded
dataset (see [training.md](training.md#supported-data-formats) for how
data is loaded) and picks one of four tasks:

| Dataset shape | Detected task |
|---|---|
| Plain text (one or more `.txt`/`.md` files, or JSON/JSONL records with a `text` field and no label) | `text-generation` |
| Text with labels (class subfolders, or JSON/JSONL with `text` + `label`) | `text-classification` |
| Tabular data whose target column is numeric with many distinct values | `regression` |
| Tabular data whose target column is categorical, or numeric with few distinct integer values | `classification` |

The target column for tabular data is chosen by looking for a column
named `label`, `target`, `class`, `category`, `y`, or `output` (case
insensitive); if none of those exist, the **last column** in the file is
used, which is a common convention in tabular datasets.

You can always override detection explicitly: `tl.train("./data",
task="regression")`.

## 2. Architecture selection

Once the task is known, `tensorless/auto/config.py` picks:

- **model type**: `transformer` for text tasks, `mlp` for tabular tasks
- **size** (`d_model`, `layers`, `heads`): scaled to dataset size, from
  a tiny 2-layer/64-dim model for a few hundred effective text examples up to an
  8-layer/384-dim model for 50,000+ examples

For text corpora, effective examples include corpus character count, so a
single large `.txt` file is not treated like one training example. BPE
vocabulary size is also bounded from corpus character diversity rather than
always using a fixed oversized vocabulary.

This is a heuristic, not a search — the goal is "a model that trains
quickly and doesn't wildly overfit or underfit for typical dataset
sizes," not the best possible architecture. Override any of it:
`tl.train("./data", d_model=512, layers=6)`.

For a packaged English grammar starter corpus, use
`tl.pretrain(out="english.tl")`. It is intended for demos and smoke tests;
larger local corpora should be passed to `tl.train()`.

## 3. Hyperparameter selection

Batch size, epoch count, learning rate, warmup steps, and the
validation split are all similarly scaled to dataset size. See
[configuration.md](configuration.md) for the exact defaults and how to
override each one.

## 4. Hardware selection

`tensorless/devices/device.py` picks, in order: TPU (if `torch_xla` is
installed and usable) → CUDA GPU (if available) → Apple MPS (if
available) → CPU. Precision is chosen alongside it: `bf16` on TPU,
`bf16` or `fp16` on GPU depending on hardware support, and `fp32`
everywhere else for stability.

This selection isn't blind — Tensorless actually checks each backend is
usable (not just importable) before choosing it, and if a device you
explicitly request isn't available, it downgrades gracefully rather
than erroring.

## 5. The Smart Auto Check

This is the part that makes repeated `tl.train("./data")` calls safe and
cheap. Every dataset gets a **fingerprint**: a hash of every file's
content, size, and relative path under the given directory (see
`tensorless/data/fingerprint.py`). This fingerprint is content-based, not
timestamp-based, so touching a file without changing it, or copying a
dataset to a new machine, doesn't trigger a false "changed" signal.

When you call `tl.train(path, out="model.tl")`, Tensorless checks, in
order:

1. **Does `model.tl` already exist, with a fingerprint matching the
   current dataset, and `training_complete=True`?**
   → Return it immediately. No training happens.

2. **Is there a checkpoint at `model.tl.ckpt/` whose fingerprint matches
   the current dataset?**
   - If that checkpoint's training wasn't finished
     (`training_complete=False`) → **resume** from it.
   - If it was actually finished but the final `.tl` file is missing
     (e.g. the process died right after the last checkpoint write, before
     the `.tl` file could be written) → package that checkpoint into
     `model.tl` directly, no retraining needed.

3. **Is there a checkpoint whose fingerprint does *not* match?** → The
   dataset changed since that checkpoint was created.
   - By default, Tensorless retrains from scratch automatically (and
     prints a message explaining why).
   - Pass `ask_on_data_change=True` to instead raise a `ConfigError`
     and let you decide, rather than silently retraining.

4. **None of the above?** → Train from scratch.

`force=True` skips all of this and always retrains from scratch,
clearing any existing checkpoint first.

See [checkpointing.md](checkpointing.md) for what's actually inside a
checkpoint and exactly how resumption reconstructs training state.
