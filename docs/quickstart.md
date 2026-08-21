# Quick Start

## 1. Get some data

Any of these work:

- a `.txt` file (Tensorless trains a text-generation model on it)
- a directory of `.txt` files organized into class subfolders, e.g.
  `data/positive/*.txt`, `data/negative/*.txt` (text classification)
- a `.csv` file with a target column, e.g. `label` or `price` (tabular
  classification or regression)
- `.json` / `.jsonl` files, either free-text records (`{"text": "..."}`)
  or structured records with a target field

If you're not sure what Tensorless will do with your data, ask it:

```python
import tensorless as tl
tl.inspect("./data")
```

This prints the detected task, dataset size, and any warnings —
without training anything.

## 2. Train

```python
tl.train("./data")
```

This one line:

1. loads and inspects your dataset
2. detects the task (text generation, text classification,
   classification, or regression)
3. picks a model architecture and size appropriate for your data
4. picks an optimizer, learning rate, batch size, and epoch count
5. picks the best available hardware (TPU → GPU → CPU)
6. trains with validation and early stopping
7. checkpoints periodically so it can resume if interrupted
8. saves everything to a single `model.tl` file

## 3. Use the model

For a quick interactive check from the command line:

```bash
tensorless run model.tl
```

For text-generation models with no `--prompt`, this drops you into an
interactive chat loop.

From Python:

```python
model = tl.load("model.tl")

# text-generation
print(model.generate("Once upon a time"))

# text-classification / tabular classification / regression
print(model.predict("this movie was great"))
print(model.predict({"age": 30, "income": 90000}))
```

## 4. Run it again

Nothing changed? Tensorless notices and skips retraining:

```python
tl.train("./data")  # instant -- returns the existing model.tl
```

Changed your data? It retrains automatically:

```python
tl.train("./data")  # data changed -- retrains
```

Training got interrupted (crash, Ctrl+C, out of time on a shared
cluster)? Just call it again:

```python
tl.train("./data")  # resumes from the last checkpoint
```

This is the **Smart Auto Check** — see
[automatic_mode.md](automatic_mode.md) for exactly how it decides.

## Next steps

- [Beginner Tutorial](tutorial.md) — a slower, guided walkthrough with
  explanations at every step
- [Training](training.md) — the full `tl.train()` reference
- [Configuration](configuration.md) — every setting you can override
