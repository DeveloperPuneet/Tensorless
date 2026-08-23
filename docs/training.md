# Training

## Basic usage

```python
import tensorless as tl

tl.train("./data")
```

Returns a `LoadedModel` (see [inference.md](inference.md)) ready for
predictions, and writes `model.tl` (plus a `model.tl.ckpt/` checkpoint
directory) to the current directory.

## Pretraining and fine-tuning

Use the built-in corpus (or any text corpus) to create a base model, then
continue training its learned weights on your own data:

```python
base = tl.pretrain(out="english_pretrained.tl", epochs=10)
model = tl.train(
    "./my_text.txt",
    pretrained="english_pretrained.tl",
    out="my_model.tl",
    epochs=5,
    learning_rate=1e-4,
)
```

The source tokenizer and compatible architecture are reused automatically.
The fine-tuning run starts a fresh optimizer and scheduler; interrupted
fine-tuning still resumes from its own checkpoint. Use
`tl.load_pretrained("english_pretrained.tl")` when you want to load the base
model directly for inference.

## Supported data formats

| Format | Notes |
|---|---|
| `.txt`, `.md` | Treated as raw text for language modeling |
| `.csv`, `.tsv` | Tabular; first row is the header |
| `.json` | A list of records, or `{"data": [...]}` / `{"records": [...]}` |
| `.jsonl`, `.ndjson` | One JSON object per line |
| A directory of the above | Merged; see below |
| A directory of class subfolders of `.txt`/`.md` files | Text classification, folder name = label |

For JSON/JSONL records, Tensorless looks for a field named `text`,
`content`, `body`, `document`, or `sentence` to treat as the input text,
and `label`, `target`, `class`, `category`, or `y` for the label if
present. If a text field is found but no label field, it's a
text-generation dataset. If both are found, it's text classification.

For CSV/TSV/JSON/JSONL without a text field, records are treated as
tabular data (see [automatic_mode.md](automatic_mode.md) for how the
target column and task type are chosen).

A directory can mix multiple files of the *same* format (e.g. several
`.csv` files, or several `.txt` files) — they're concatenated. Mixing
plain text files with structured (JSON/CSV) files in the same directory
raises a `DataError` asking you to separate them.

For tabular data, numeric columns are robustly scaled and missing values use
the training median. ISO-8601 date and datetime columns are converted to
numeric timestamps. Categorical columns are frequency-ranked and capped at
1,000 learned values; rare or unseen values use the `<unk>` category.

Tensorless never modifies, moves, or deletes files in your dataset
directory. It only ever reads from `path`; all output goes to the `out`
file and `checkpoint_dir`.

## Overriding auto-configuration

Every field of `TrainConfig` can be passed as a keyword argument. A few
common ones:

```python
tl.train(
    "./data",
    task="text-generation",       # skip auto-detection
    d_model=512, layers=6, heads=8,
    batch_size=32, epochs=20,
    learning_rate=3e-4,
    val_split=0.15,
    device="cuda",
    out="my_model.tl",
)
```

See [configuration.md](configuration.md) for the full list.

## Validation and early stopping

By default, Tensorless holds out `val_split` of the data (10% for
datasets with 50+ examples, 0% for smaller ones where a held-out split
wouldn't be meaningful) and tracks validation loss after each epoch. If
validation loss doesn't improve by at least `min_delta` for `patience`
consecutive epochs, training stops early. The automatic default is 5
consecutive epochs. Before the completed model is written to the `.tl` output
file, Tensorless restores the weights from the epoch with the best validation
loss. This applies equally to regular training and built-in pretraining.

## Checkpointing during training

A checkpoint is written every `checkpoint_every` steps (default: 50)
and at the end of every epoch, to `<out>.ckpt/checkpoint.pt`. See
[checkpointing.md](checkpointing.md) for what's in it and how it's used
for resumption.

## What each task trains

| Task | Model | What's learned |
|---|---|---|
| `text-generation` | Small GPT-style decoder transformer, BPE tokenizer by default | Next-token prediction over your text |
| `text-classification` | Same transformer backbone, BPE tokenizer by default, classification head on the final token | Text → one of your labeled classes |
| `classification` | MLP with per-column categorical embeddings | Row of features → one of your labeled classes |
| `regression` | Same MLP, single continuous output | Row of features → a number |

## Resuming and force-retraining

```python
tl.train("./data")               # resumes an interrupted run automatically
tl.train("./data", force=True)   # always retrain from scratch
```

See [automatic_mode.md](automatic_mode.md#5-the-smart-auto-check) for
the exact decision logic.

## Command-line equivalent

```bash
tensorless train ./data --d-model 512 --layers 6 --batch-size 32
```

See [cli.md](cli.md) for the full CLI reference.
