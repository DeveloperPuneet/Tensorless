# Contributing

Thanks for considering contributing to Tensorless. This is an early-
stage project, so there's plenty of room to shape it.

## Setup

```bash
git clone https://github.com/tensorless/tensorless.git
cd tensorless
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

The suite covers data loading, task detection, fingerprinting, training
for all four supported tasks, checkpoint/resume behavior, the Smart Auto
Check, `.tl` serialization, and the CLI. Training tests use tiny models
and datasets (a handful of epochs, small dimensions) so the full suite
runs in a few minutes on CPU.

If you're adding a feature, please add tests in the corresponding
`tests/test_*.py` file (or a new one) rather than only testing manually.

## Code style

- Keep modules focused — see [architecture.md](architecture.md) for
  where things belong. If you find yourself adding unrelated
  responsibilities to an existing file, it probably wants a new module.
- Every user-facing error should raise a `TensorlessError` subclass
  (see `errors.py`) with a message that explains what went wrong and,
  where possible, what to do about it.
- Prefer explicit, inspectable heuristics over opaque logic for anything
  in the auto-configuration path — see "Design principles" in
  [architecture.md](architecture.md).
- Docstrings on public functions/classes should explain *why*, not just
  restate the signature.

## Areas that could use help

See [roadmap.md](roadmap.md) for planned work. A few good starting
points:

- **New data formats**: Parquet, Excel, images, audio
- **New model types**: proper BPE/subword tokenization as an
  alternative to the default char-level tokenizer; CNNs; larger
  pretrained-backbone fine-tuning
- **New backends**: JAX or a lighter pure-NumPy backend for environments
  without PyTorch
- **`.tl` format migration**: forward-compatible loading of older
  format versions
- **Better auto-configuration**: replacing the current size-based
  heuristics with something that also looks at data complexity (e.g.
  vocabulary size, class balance)

## Submitting changes

1. Fork the repo and create a branch for your change.
2. Add or update tests covering the change.
3. Run the full test suite and make sure it passes.
4. Update relevant docs in `docs/` — a feature without documentation
   isn't done, per this project's own stated principles.
5. Open a pull request describing what changed and why.

## Reporting bugs

Please include:
- Tensorless version
- A minimal reproduction (smallest dataset + `tl.train(...)` call that
  shows the problem)
- The full error message/traceback

## Code of conduct

Be respectful, assume good faith, and keep discussion focused on the
technical merits of a change.
