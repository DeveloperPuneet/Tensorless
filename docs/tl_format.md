# The `.tl` Format

A `.tl` file is a single, portable file containing everything needed to
run a trained model on a different machine, with no access to the
original training data or code beyond having Tensorless installed.

## Structure

Under the hood, `.tl` is a `torch.save`/`torch.load`-compatible pickle
archive (see `tensorless/serialization/tl_format.py`) containing a
dictionary:

```python
{
    "tl_format_version": 1,
    "tensorless_version": "0.1.0",
    "task": "text-generation",         # or text-classification / classification / regression
    "model_type": "transformer",       # or mlp
    "config": { ... },                  # full resolved TrainConfig used
    "meta": { ... },                    # vocab_size / n_classes / column info -- whatever the model needs to rebuild
    "model_state_dict": { ... },        # PyTorch model weights
    "tokenizer_state": { ... } | None,  # CharTokenizer vocab, for text tasks
    "preprocessor_state": { ... } | None,  # TabularPreprocessor state, for tabular tasks
    "dataset_fingerprint": "...",       # fingerprint of the training dataset
    "training_complete": True,
    "metrics": { ... },                 # final train/val loss, step count, wall-clock time
}
```

## Why a single file

The requirement driving this design: you should be able to `scp` or
email one `model.tl` file and have someone else load and run it, without
also sending them a directory of auxiliary tokenizer/config files. That
means every piece of state a trained model needs — architecture,
weights, and the exact preprocessing pipeline used to produce its
inputs — has to be embedded together.

## Loading

```python
model = tl.load("model.tl")
```

internally calls `tensorless.serialization.tl_format.load_tl(path)`,
which:

1. Reads the pickle archive
2. Verifies all required fields are present (raises `SerializationError`
   with a clear message if the file is corrupt or missing fields)
3. Verifies the file's format version isn't newer than what the
   installed Tensorless version supports (raises `SerializationError`
   asking you to upgrade, rather than silently misbehaving)

Then `LoadedModel.__init__` (in `tensorless/runtime.py`) uses `task`,
`model_type`, `config`, and `meta` to rebuild the exact same model class
and shape, and loads `model_state_dict` into it.

## Versioning

`tl_format_version` (currently `1`) is bumped whenever the payload
structure changes in a backward-incompatible way. Tensorless refuses to
load files from a newer format version than it understands, rather than
guessing at a schema it doesn't recognize. Migrating old-format files
forward is on the [roadmap](roadmap.md) but not implemented yet — see
[limitations.md](limitations.md).

## Inspecting a `.tl` file without loading the model

```bash
tensorless info model.tl
```

```json
{
  "task": "text-generation",
  "model_type": "transformer",
  "tensorless_version": "0.1.0",
  "tl_format_version": 1,
  "training_complete": true,
  "metrics": {...},
  "dataset_fingerprint": "f6e1539b96a467ff"
}
```
