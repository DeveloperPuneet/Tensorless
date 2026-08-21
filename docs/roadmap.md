# Roadmap

Tensorless is early-stage. This is a snapshot of planned direction, not
a commitment or timeline.

## Near-term

- **Subword/BPE tokenization** as an alternative to the default
  character-level tokenizer, for better efficiency on larger text
  corpora
- **`.tl` format migration** — forward-compatible loading of files
  written by older Tensorless versions
- **Better tabular preprocessing** — handling of date/datetime columns,
  high-cardinality categorical columns, and more robust outlier handling
  for regression targets
- **Progress bars** for training (currently plain print-based logging)
- **`tl.train(..., resume=False)`** enforcement — currently `resume` is
  accepted in `TrainConfig` but the automatic resume decision doesn't
  yet fully respect an explicit `False` override in every code path
- **Multi-GPU / distributed training** for larger datasets

## Medium-term

- **Additional model families**: CNNs for structured sequence/image-like
  data, fine-tuning of pretrained backbones rather than always training
  from scratch
- **Additional data formats**: Parquet, Excel (`.xlsx`), image
  directories, audio
- **Hyperparameter search mode**: an opt-in `tl.train("./data",
  search=True)` that tries a small set of configurations and keeps the
  best, rather than a single heuristic choice
- **Data quality auto-fixes**: currently `tl.inspect()` only *reports*
  problems like missing values or class imbalance; a future mode could
  offer to fix them (with explicit user opt-in, consistent with "never
  silently modify user data")

## Long-term / exploratory

- **Alternate backends** (JAX, a lightweight NumPy-only backend) behind
  the same `tl.train()`/`tl.load()` API
- **Streaming/out-of-core training** for datasets too large to fit in
  memory
- **Export to other formats** (ONNX, TorchScript) from a `.tl` file for
  deployment outside Python

## Explicitly not planned

See [limitations.md](limitations.md) for things that are out of scope
by design rather than just "not built yet."
