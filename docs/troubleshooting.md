# Troubleshooting

## Installation issues

**`ModuleNotFoundError: No module named 'numpy'`**
NumPy is installed automatically with `pip install -e .`. If it is missing,
install it with `python -m pip install numpy`, then reinstall Tensorless.

**`tensorless: command not found` after installing**
Make sure the Python environment's `bin`/`Scripts` directory is on your
`PATH`, or run the CLI as `python -m tensorless.cli.main` instead.

## Training issues

**`DataError: Path '...' does not exist.`**
Double-check the path — it's relative to your current working
directory, not the dataset's own location.

**`DataError: No supported data files found in '...'.`**
Your directory doesn't contain any `.txt`, `.md`, `.json`, `.jsonl`,
`.csv`, or `.tsv` files. See [training.md](training.md#supported-data-formats).

**`DataError: Directory '...' mixes plain text files with structured (json/csv) files.`**
Keep one data format per directory — split text files and
structured files into separate folders and train separately, or convert
one format into the other.

**`DataError: '<file>' is not valid UTF-8 text.`**
Tensorless expects UTF-8 text files. Re-save the file with UTF-8
encoding (most editors have an "encoding" option in the save dialog).

**Training seems too slow**
- Confirm the device being used: check the printed
  `[tensorless] task=... device=...` line, or `model.config["device"]`
  after training.
- On CPU, try a smaller model: `tl.train("./data", d_model=64, layers=2)`.
- For quick experimentation, cap the run with `max_steps=`.

**Loss is `nan` or not decreasing**
- Try a lower learning rate: `tl.train("./data", learning_rate=1e-4)`.
- Make sure `grad_clip` isn't disabled (`grad_clip=0`) if your data has
  outliers.
- For regression, extreme target value ranges can cause instability;
  Tensorless standardizes numeric targets automatically, but very heavy-
  tailed distributions (a few huge outliers) can still be tricky —
  consider removing or capping extreme outliers in your data first.

## Smart Auto Check issues

**`tl.train()` isn't retraining even though I changed my code (not my data)**
This is expected — the Smart Auto Check only looks at *dataset*
content, not your training script. Changing hyperparameters in code
without changing the data won't trigger a retrain; the dataset
fingerprint is unchanged, so the existing model is reused. Pass
`force=True` to retrain with new hyperparameters, or delete `model.tl`
and its `.ckpt` directory.

**`tl.train()` retrained when I didn't expect it to**
Something under your dataset path changed — even a file's *content*
changing while its filename stays the same will change the fingerprint.
If you regenerate the same logical dataset from a script (e.g. with a
different random seed each run), the fingerprint will differ every time.
Fix the seed in your data-generation script if you want a stable
fingerprint across runs.

## Checkpoint / resume issues

**`CheckpointError: Checkpoint at '...' is corrupt or incompatible.`**
The checkpoint file was likely truncated by an interruption during the
  (non-atomic part of the) write, or created by an incompatible Tensorless
  version. Delete the `.ckpt` directory and retrain — you'll lose progress
on the interrupted run, but the checkpoint being unreadable means it
can't be safely resumed regardless.

**Resume used the wrong hyperparameters**
This is expected: resuming always uses the checkpoint's original config,
ignoring any new overrides you pass, because the model architecture must
match the saved weights exactly. See
[checkpointing.md](checkpointing.md#how-resumption-works).

## `.tl` file issues

**`SerializationError: '...' was created with a newer .tl format...`**
Upgrade Tensorless: `pip install -e . --upgrade` (from an updated
checkout).

**`SerializationError: '...' is missing required field(s) [...]`**
The file is corrupt, truncated, or not actually a Tensorless `.tl` file.

## Getting more help

If you hit something not covered here, please open an issue with:
- the full error message and traceback
- your Tensorless version (`python -c "import tensorless; print(tensorless.__version__)"`)
- a minimal reproduction if possible (a tiny dataset + the exact
  `tl.train(...)` call)
