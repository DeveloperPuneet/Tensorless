# Tensorless

Tensorless trains small custom models with sensible defaults. It uses a native
NumPy engine on CPU and optional JAX or MLX backends for accelerator execution.
It supports
text generation, text classification, tabular classification, and regression.

## Install

```bash
pip install -e .
```

Optional accelerator backends:

```bash
pip install -e '.[cuda]'   # JAX CUDA
pip install -e '.[tpu]'    # JAX TPU
pip install -e '.[mps]'    # Apple Silicon MLX
```

CUDA and TPU backends currently accelerate transformer text tasks. Tabular
tasks and unsupported platforms use the native CPU engine.

## Train on your data

```python
import tensorless as tl

model = tl.train("./corpus.txt", task="text-generation")
print(model.generate("The", max_new_tokens=40))
```

Text files are trained as next-token language models. BPE is the default
tokenizer; use `tokenizer="char"` for a character-level model. Tensorless
derives model size, batch size, epochs, validation, device, and BPE vocabulary
size from the data, while every setting can be overridden.

Long text is tokenized lazily and fed through the native engine in fixed-size batches.
The automatic batch size uses a token budget; reduce `batch_size` if your
available memory is limited.

## English starter pretraining

```python
import tensorless as tl

model = tl.pretrain(out="english.tl", epochs=20, max_seq_len=128)
print(model.generate("A complete sentence", max_new_tokens=30))
```

This offline starter corpus contains English prose and grammar examples. It is
for demos and smoke tests, not a replacement for a large language dataset. For
real pretraining, pass your own `.txt` corpus to `tl.train()` and increase the
training settings as your hardware allows.

## Other tasks

```python
tl.train("reviews/", task="text-classification")
tl.train("housing.csv", task="regression")
```

Tabular preprocessing automatically handles numeric values, ISO dates, and
high-cardinality categories. Missing and rare values are handled using the
fitted training data, and the same preprocessing is stored in the `.tl` file.

Models are saved as `.tl` files and can be loaded later:

```python
model = tl.load("model.tl")
print(model.info())
```

See the [documentation](docs/quickstart.md) for data formats, configuration,
checkpointing, and the command-line interface.