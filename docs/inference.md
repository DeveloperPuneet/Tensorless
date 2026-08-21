# Inference

## Loading a model

```python
import tensorless as tl

model = tl.load("model.tl")
```

`model` is a `LoadedModel`. It rebuilds the exact architecture used at
training time from the `.tl` file's embedded config, loads the trained
weights, and moves everything to the requested device (default: the
device recorded at training time, downgraded to CPU if that device
isn't available on the current machine).

```python
model = tl.load("model.tl", device="cpu")  # force CPU regardless of training device
```

## Text generation

```python
model.generate("Once upon a time", max_new_tokens=200, temperature=0.8, top_k=40)
```

- `temperature`: higher = more random, lower = more deterministic
- `top_k`: only sample from the top-k most likely next tokens

For an interactive terminal chat loop:

```python
model.chat()
```

or from the shell:

```bash
tensorless run model.tl
```

## Text classification

```python
model.predict("this movie was great")
# -> "positive"

model.predict(["great film", "terrible film"])
# -> ["positive", "negative"]
```

## Tabular classification / regression

Pass a dict (or list of dicts) with the same column names used at
training time (excluding the target column):

```python
model.predict({"age": 35, "income": 95000, "city": "NYC"})
# classification -> "1"
# regression     -> 349213.5
```

Missing columns are imputed the same way they were during training
(numeric: training-set mean; categorical: a `<missing>` token).
Categories never seen during training map to an `<unk>` token rather
than raising an error.

## `tl.run()` — the CLI-friendly shortcut

```python
tl.run("model.tl")                    # text-generation -> interactive chat
tl.run("model.tl", prompt="hello")    # text-generation -> one generation
tl.run("sentiment.tl", prompt="great") # text-classification -> one prediction
```

For tabular models, `tl.run()` can't accept structured input as a single
string, so it points you to `tl.load(...).predict({...})` instead.

## Inspecting a loaded model

```python
model.info()
```

```python
{
    "task": "text-generation",
    "model_type": "transformer",
    "tensorless_version": "0.1.0",
    "tl_format_version": 1,
    "config": {...},          # the full resolved training config
    "metrics": {...},         # final train/val loss, step count, training time
    "training_complete": True,
    "n_parameters": 412673,
}
```

## Portability

A `.tl` file is self-contained: model weights, architecture, tokenizer
or preprocessor, and config all live in one file. You can copy it to a
different machine (with Tensorless installed) with no access to the
original dataset and it will load and run identically. See
[tl_format.md](tl_format.md) for exactly what's inside.
