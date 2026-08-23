# Installation

## Requirements

- Python 3.9 or later
- NumPy 1.24 or later (installed automatically as a dependency)
- Tensorless always runs its native vectorized engine, on CPU by default.
  Installing the optional accelerator extras below lets it detect and use a
  GPU/TPU automatically, for both transformer and tabular MLP models.

## GPU / TPU / Apple Silicon support

By default `pip install tensorless` only pulls in NumPy, so even on a
machine with a GPU, device auto-detection will correctly report that no
accelerator backend is *installed* and fall back to CPU. To actually train
and run on your hardware, install the matching extra:

```bash
pip install 'tensorless[cuda]'   # NVIDIA GPUs, via JAX
pip install 'tensorless[tpu]'    # Google TPUs, via JAX
pip install 'tensorless[mps]'    # Apple Silicon, via MLX
```

Once installed, `tl.train(...)` auto-detects the best available device
(`tpu` > `cuda` > `mps` > `cpu`) and trains on it without any extra
configuration; you can also force a specific device with
`tl.train(..., device="cuda")`.

## Verify your installation

```bash
python -c "import tensorless as tl; print(tl.__version__)"
tensorless --help
```

You should see a version string printed and the CLI's help text.

## Runtime support

Training and inference resolve the device automatically: the best available
accelerator (`tpu` > `cuda` > `mps`) is used if its extra is installed and
the hardware is detected, otherwise Tensorless falls back to the NumPy CPU
engine. You can always force a specific device with `device="cpu"`,
`device="cuda"`, etc. in `tl.train(...)`.

## Troubleshooting installation

See [troubleshooting.md](troubleshooting.md#installation-issues) for
common installation problems.
