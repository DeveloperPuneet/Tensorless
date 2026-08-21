# Examples

Runnable, tested scripts live in [`examples/`](../examples) — each one
generates its own tiny sample dataset if it doesn't already exist, so
you can run them immediately with no setup:

```bash
python examples/text_generation_example.py
python examples/text_classification_example.py
python examples/tabular_classification_example.py
python examples/tabular_regression_example.py
```

The snippets below are shortened excerpts of what each script does.

## Text generation

```python
import tensorless as tl

tl.train("examples/data/tiny_corpus.txt", out="model.tl")
model = tl.load("model.tl")
print(model.generate("the quick", max_new_tokens=60))
```

## Text classification

```python
import tensorless as tl

# expects examples/data/reviews/positive/*.txt and .../negative/*.txt
model = tl.train("examples/data/reviews", out="sentiment.tl")
print(model.predict("this movie was wonderful and I really enjoyed watching it"))
```

## Tabular classification

```python
import tensorless as tl

model = tl.train("examples/data/customers.csv", out="churn.tl")
print(model.predict({"age": 35, "income": 95000, "city": "NYC"}))
```

## Tabular regression

```python
import tensorless as tl

model = tl.train("examples/data/houses.csv", out="prices.tl")
print(model.predict({"sqft": 2000, "bedrooms": 3, "city": "NYC"}))
```

## Overriding auto-configuration for a bigger model

```python
import tensorless as tl

tl.train(
    "./data",
    d_model=512,
    layers=6,
    heads=8,
    batch_size=32,
    learning_rate=3e-4,
    epochs=30,
)
```

## Explicit device selection

```python
import tensorless as tl

tl.train("./data", device="cuda")   # force GPU
tl.train("./data", device="cpu")    # force CPU even if a GPU is present
```

## Handling errors gracefully

```python
import tensorless as tl

try:
    model = tl.train("./data")
except tl.DataError as e:
    print(f"Problem with your dataset: {e}")
except tl.ConfigError as e:
    print(f"Problem with your configuration: {e}")
```

## Checking a dataset before committing to training

```python
import tensorless as tl

report = tl.inspect("./data")
if report.n_examples < 50:
    print("Warning: this is a small dataset, consider collecting more data")
if report.warnings:
    for w in report.warnings:
        print("!", w)
```

## CLI equivalents

```bash
tensorless inspect ./data
tensorless train ./data --out model.tl --epochs 20
tensorless run model.tl
tensorless info model.tl
```
