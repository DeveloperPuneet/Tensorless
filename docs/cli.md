# Command-Line Interface

Installing Tensorless (`pip install -e .`) puts a `tensorless` command
on your `PATH`.

## `tensorless train`

```bash
tensorless train <path> [options]
```

| Option | Description |
|---|---|
| `--out PATH` | Output `.tl` path (default: `model.tl`) |
| `--force` | Retrain even if a matching model/checkpoint exists |
| `--d-model N` | Hidden dimension |
| `--layers N` | Number of layers |
| `--heads N` | Attention heads |
| `--batch-size N` | Batch size |
| `--epochs N` | Max epochs |
| `--learning-rate F` | Learning rate |
| `--device {cpu,cuda,tpu,mps}` | Force a device |
| `--quiet` | Suppress training logs |

For any configuration option not exposed as a CLI flag, use the Python
API — the CLI covers the common cases; `tl.train()` covers everything in
[configuration.md](configuration.md).

Example:

```bash
tensorless train ./data --out sentiment.tl --epochs 20 --batch-size 32
```

## `tensorless run`

```bash
tensorless run <model.tl> [--prompt TEXT]
```

- Text-generation model, no `--prompt`: starts an interactive chat.
- Text-generation model, with `--prompt`: prints one generated
  continuation.
- Text-classification model, with `--prompt`: prints the predicted
  class.
- Tabular (classification/regression) model: the CLI can't accept
  structured input as a single string, so it prints a pointer to the
  Python API (`tl.load(path).predict({...})`).

## `tensorless inspect`

```bash
tensorless inspect <path>
```

Loads the dataset, runs task detection, and prints a report: detected
kind and task, example count, columns (for tabular data), and any
warnings/recommendations — without training anything.

## `tensorless info`

```bash
tensorless info <model.tl>
```

Prints a JSON summary of a trained model: task, model type, versions,
whether training completed, final metrics, and a truncated dataset
fingerprint.

```json
{
  "task": "classification",
  "model_type": "mlp",
  "tensorless_version": "0.1.0",
  "tl_format_version": 1,
  "training_complete": true,
  "metrics": {"final_train_loss": 0.42, "final_val_loss": 0.51, "global_step": 96, "elapsed_seconds": 0.14},
  "dataset_fingerprint": "de779ad33e2dc1d4"
}
```

## Exit codes

All commands return `0` on success. Errors raised as `TensorlessError`
subclasses (bad data, bad config, corrupt checkpoint/model, etc.) are
caught, printed to stderr as `tensorless: error: <message>`, and result
in exit code `1`.
