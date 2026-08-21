# Roadmap

Tensorless is early-stage. This is a snapshot of planned direction, not
a commitment or timeline.

## Near-term

- **Progress bars** for training (currently plain print-based logging)
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
- **Export to other formats** (ONNX, TorchScript) from a `.tl` file for
  deployment outside Python

## Explicitly not planned

See [limitations.md](limitations.md) for things that are out of scope
by design rather than just "not built yet."
