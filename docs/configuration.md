# Configuration Reference

Every one of these can be passed as a keyword argument to `tl.train()`.
Anything left unset (`None`) is chosen automatically — see
[automatic_mode.md](automatic_mode.md) for the heuristics behind the
defaults.

## Output / lifecycle

| Field | Default | Description |
|---|---|---|
| `out` | `"model.tl"` | Where to save the trained model |
| `force` | `False` | Retrain from scratch even if an up-to-date model/checkpoint exists |
| `resume` | auto | Force (`True`) or forbid (`False`) resuming from a checkpoint |
| `ask_on_data_change` | `False` | Raise `ConfigError` instead of auto-retraining when the dataset has changed |

## Task / architecture

| Field | Default | Description |
|---|---|---|
| `task` | auto-detected | `"text-generation"`, `"text-classification"`, `"classification"`, or `"regression"` |
| `model_type` | auto | `"transformer"` or `"mlp"` |
| `d_model` | auto (scaled to data size) | Hidden dimension |
| `layers` | auto | Number of transformer blocks / MLP hidden layers |
| `heads` | auto | Attention heads (transformer only) |
| `ff_mult` | auto | Feed-forward expansion multiplier (transformer only) |
| `dropout` | `0.1` | Dropout probability |
| `gradient_checkpointing` | `False` | Recompute native transformer block activations during backward to reduce memory use |
| `max_seq_len` | `256` (text) / `1` (tabular) | Max sequence length in tokens (text tasks) |
| `tokenizer` | `"bpe"` | Text tokenizer: `"bpe"` or `"char"` |
| `bpe_vocab_size` | `1000` | Maximum vocabulary size when `tokenizer="bpe"` |

## Optimization

| Field | Default | Description |
|---|---|---|
| `optimizer` | `"adamw"` | `"adamw"`, `"adam"`, or `"sgd"` |
| `learning_rate` | `3e-4` (transformer) / `1e-3` (mlp) | Peak learning rate |
| `weight_decay` | `0.01` | L2 weight decay |
| `batch_size` | auto (scaled to data size, 8-64) | Training batch size |
| `epochs` | auto (scaled to data size, 5-40) | Max epochs |
| `max_steps` | `None` | If set, stop after this many optimizer steps regardless of epoch count |
| `grad_clip` | `1.0` | Gradient norm clipping threshold |
| `warmup_steps` | auto | Linear LR warmup steps |

## Validation / early stopping

| Field | Default | Description |
|---|---|---|
| `val_split` | `0.1` (if ≥50 examples, else `0`) | Fraction of data held out for validation |
| `patience` | `3` | Epochs without improvement before stopping early |
| `min_delta` | `1e-4` | Minimum improvement to count as "improved" |

## Hardware

| Field | Default | Description |
|---|---|---|
| `device` | auto (`tpu` > `cuda` > `mps` > `cpu`) | Force a specific device; accelerator backends require their optional package |
| `precision` | auto | `"fp32"`, `"fp16"`, or `"bf16"`; JAX accelerator transformers use the selected dtype and compute loss in fp32 |

## Checkpointing

| Field | Default | Description |
|---|---|---|
| `checkpoint_every` | `50` | Steps between checkpoint writes |
| `checkpoint_dir` | `"<out>.ckpt"` | Checkpoint directory |
| `checkpoint_shard_size_mb` | `0` | Split model and optimizer checkpoint state into parts of approximately this size; `0` keeps the single-file format |

## Misc

| Field | Default | Description |
|---|---|---|
| `seed` | `42` | Random seed |
| `verbose` | `True` | Print training progress |

## Example: fully manual configuration

```python
tl.train(
    "./data",
    task="text-generation",
    model_type="transformer",
    d_model=512, layers=6, heads=8, ff_mult=4, dropout=0.1,
    max_seq_len=256,
    optimizer="adamw", learning_rate=3e-4, weight_decay=0.01,
    batch_size=32, epochs=20, grad_clip=1.0, warmup_steps=200,
    val_split=0.1, patience=5, min_delta=1e-4,
    device="cuda", precision="bf16",
    checkpoint_every=100,
    out="my_model.tl",
    seed=0,
)
```

Passing an unrecognized keyword raises a `ConfigError` listing every
valid field name, so typos are caught immediately rather than silently
ignored.
