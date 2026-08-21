# Installation

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later (installed automatically as a dependency)
- Optional: a CUDA-capable GPU, or a TPU with `torch_xla` installed, for
  faster training. Tensorless works fine on CPU-only machines too.

## Install from source

Tensorless isn't published to PyPI yet. Install it directly from a
checkout of this repository:

```bash
git clone https://github.com/tensorless/tensorless.git
cd tensorless
pip install -e .
```

The `-e` (editable) flag means changes to the source are picked up
immediately without reinstalling — useful if you're also contributing.

If you plan to run the test suite, install the dev extras too:

```bash
pip install -e ".[dev]"
```

## Verify your installation

```bash
python -c "import tensorless as tl; print(tl.__version__)"
tensorless --help
```

You should see a version string printed and the CLI's help text.

## GPU / TPU support

Tensorless detects available hardware automatically — no extra
configuration needed on your part. What it detects depends on what's
installed in your environment:

- **CUDA GPUs**: detected automatically if `torch.cuda.is_available()`
  returns `True`, which normally means you installed a CUDA-enabled
  build of PyTorch matching your GPU driver. See
  [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for
  the right install command for your system.
- **Apple Silicon (MPS)**: detected automatically on macOS with an
  M-series chip, via `torch.backends.mps`.
- **TPU**: requires `torch_xla` to be installed separately (this is
  typically pre-installed in TPU-enabled cloud environments like Google
  Colab TPU runtimes or GCP TPU VMs).

If none of these are available, Tensorless silently falls back to CPU —
you never need to configure this yourself, though you can force a
specific device with `tl.train(..., device="cpu")` if you want to.

## Troubleshooting installation

See [troubleshooting.md](troubleshooting.md#installation-issues) for
common installation problems.
