# Architecture

This page is for people extending or contributing to Tensorless, not
end users.

## Package layout

```
tensorless/
├── __init__.py           public API surface: train, run, load, inspect
├── api.py                 orchestration for train()/run(): Smart Auto Check + wiring
├── config.py               TrainConfig (user-facing) / ResolvedConfig (fully resolved)
├── errors.py                exception hierarchy
├── runtime.py               LoadedModel: rebuilds a model from a .tl payload for inference
├── _version.py               package + .tl format version numbers
├── data/
│   ├── loader.py              path -> Dataset (txt/json/jsonl/csv/dirs)
│   ├── fingerprint.py          content-based dataset hashing
│   ├── inspector.py             tl.inspect() report generation
│   └── tabular.py                 TabularPreprocessor: numeric/categorical encoding
├── auto/
│   ├── detector.py               Dataset -> task string
│   └── config.py                   Dataset + TrainConfig -> ResolvedConfig
├── tokenization/
│   ├── char_tokenizer.py          CharTokenizer: vocab build/encode/decode/save/load
│   └── bpe_tokenizer.py           BPETokenizer: corpus-trained subword encoding
├── models/
│   ├── transformer.py              TinyTransformer (GPT-style decoder)
│   ├── mlp.py                       TabularMLP
│   └── registry.py                   build_model(task, model_type, cfg, meta)
├── training/
│   ├── data_prep.py                  Dataset -> DataLoaders per task
│   ├── trainer.py                     the training loop (run_training)
│   └── early_stopping.py               EarlyStopping
├── checkpoint/
│   └── manager.py                      CheckpointManager: atomic save/load/clear
├── serialization/
│   └── tl_format.py                     save_tl/load_tl: the .tl file format
├── devices/
│   └── device.py                         hardware auto-detection + torch.device resolution
└── cli/
    └── main.py                            argparse-based CLI
```

## Data flow for `tl.train("./data")`

```
path
  │
  ▼
data.loader.load_dataset(path)          -> Dataset (kind, texts/records, columns)
  │
  ▼
auto.detector.detect_task(ds)           -> "text-generation" | "text-classification"
  │                                          | "classification" | "regression"
  ▼
auto.config.resolve_config(ds, TrainConfig) -> ResolvedConfig (every field concrete)
  │
  ▼
training.data_prep.prepare_*(ds, cfg)   -> PreparedData (train/val DataLoaders,
  │                                          meta, tokenizer/preprocessor)
  ▼
models.registry.build_model(task, model_type, cfg, meta) -> nn.Module
  │
  ▼
training.trainer.run_training(...)      -> trains, checkpoints periodically,
  │                                          returns final weights + metrics
  ▼
serialization.tl_format.save_tl(out, payload) -> model.tl
  │
  ▼
runtime.LoadedModel(payload)            -> returned to the caller
```

`api.train()` wraps this whole flow with the Smart Auto Check (see
[automatic_mode.md](automatic_mode.md)): before any of the above runs,
it checks for an existing complete `.tl` file or a resumable checkpoint
matching the dataset's fingerprint.

## Design principles

- **Every automatic decision is explainable.** Auto-detection and
  auto-configuration are simple, inspectable heuristics (see
  `auto/detector.py`, `auto/config.py`), not opaque search or ML-driven
  meta-learning. Anyone reading the code can predict what Tensorless
  will choose for a given dataset.
- **Never silently touch user data.** `data/loader.py` only reads;
  nothing in the training path writes to or deletes files under the
  dataset path.
- **Fail with actionable errors.** All user-facing errors are
  `TensorlessError` subclasses with messages that say what's wrong and
  usually what to do about it (see `errors.py` and
  [troubleshooting.md](troubleshooting.md)).
- **Checkpoints are self-describing.** A checkpoint carries its own
  config, meta, and dataset fingerprint, so resuming never depends on
  external state being reconstructed correctly by the caller.
- **A `.tl` file is the unit of portability.** Nothing about inference
  should require the original dataset, training script, or checkpoint
  directory to still exist.

## Extending Tensorless

### Adding a new model type

1. Implement your `nn.Module` in `models/your_model.py`.
2. Register it in `models/registry.py`'s `build_model()`.
3. Add a branch in `training/trainer.py`'s `_compute_loss()` for how to
   compute loss for your task/model_type combination.
4. If it needs new data prep, add a `prepare_*()` function in
   `training/data_prep.py`.
5. If it changes what needs to go in the `.tl` file, extend the `meta`
   dict produced by data prep — `build_model()` receives it and can pull
   out whatever fields it needs.

### Adding a new data format

Add a branch to `data/loader.py`'s `load_dataset()` / `_load_directory()`
for the new extension, producing a `Dataset` with the appropriate `kind`.

### Adding a new backend/device

Extend `devices/device.py`'s `_*_available()` checks and
`auto_select_device()` / `get_torch_device()`.

See [contributing.md](contributing.md) for the contribution process
itself (tests, PRs, etc).
