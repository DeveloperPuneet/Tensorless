# Installation

## Requirements

- Python 3.9 or later
- NumPy 1.24 or later (installed automatically as a dependency)
- Tensorless currently runs its native vectorized engine on CPU; accelerator
  backends are planned without changing the public API.

## Verify your installation

```bash
python -c "import tensorless as tl; print(tl.__version__)"
tensorless --help
```

You should see a version string printed and the CLI's help text.

## Runtime support

The native engine uses NumPy vectorized CPU kernels and automatically resolves
the device to CPU. You can still specify `device="cpu"` explicitly in
`tl.train(...)`; the device option remains forward-compatible with future
accelerator backends.

## Troubleshooting installation

See [troubleshooting.md](troubleshooting.md#installation-issues) for
common installation problems.
