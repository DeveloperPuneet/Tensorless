# Beginner Tutorial

This tutorial walks through training three different kinds of models
with Tensorless, explaining what's happening at each step. No prior ML
experience is assumed.

## Setup

```bash
pip install -e .
mkdir tensorless_tutorial && cd tensorless_tutorial
```

## Part 1: Teach a model to write text

Create a small text file to train on:

```python
# make_data.py
lines = [f"the quick brown fox jumps over the lazy dog number {i}" for i in range(300)]
open("story.txt", "w").write("\n".join(lines))
```

```bash
python make_data.py
```

Now train:

```python
import tensorless as tl

tl.train("story.txt")
```

You'll see output like:

```
[tensorless] task=text-generation model=transformer device=cpu precision=fp32
[tensorless] epoch 1/40 train_loss=1.8421 val_loss=1.7902 (best)
...
[tensorless] training finished in 4.2s (1240 steps)
[tensorless] saved trained model to 'model.tl'
```

What happened:

- Tensorless read `story.txt`, saw it was plain text with no obvious
  structure, and decided this is a **text-generation** task.
- It built a small transformer sized for a ~16KB dataset (a few hundred
  thousand examples would get a bigger one).
- It trained with validation, stopping early once the validation loss
  stopped improving.
- It saved everything — weights, architecture, tokenizer — into
  `model.tl`.

Try it out:

```python
model = tl.load("model.tl")
print(model.generate("the quick", max_new_tokens=50))
```

Or from the terminal:

```bash
tensorless run model.tl
```

This starts an interactive chat: type a prompt, get a continuation,
repeat. Type `exit` to quit.

## Part 2: Teach a model to classify text

Organize text into class subfolders — this is how Tensorless recognizes
a text classification task:

```python
import os, random
random.seed(0)
pos = ["great", "amazing", "wonderful", "excellent"]
neg = ["terrible", "awful", "horrible", "bad"]
os.makedirs("reviews/positive", exist_ok=True)
os.makedirs("reviews/negative", exist_ok=True)
for i in range(60):
    open(f"reviews/positive/{i}.txt", "w").write(f"this was {random.choice(pos)}, loved it")
    open(f"reviews/negative/{i}.txt", "w").write(f"this was {random.choice(neg)}, hated it")
```

```python
tl.inspect("reviews")
```

```
Dataset: reviews
  detected kind : text_labeled
  detected task : text-classification
  examples      : 120
  n_classes     : 2
  classes       : ['negative', 'positive']
```

Train and predict:

```python
model = tl.train("reviews")
print(model.predict("this was wonderful, loved it"))   # "positive"
print(model.predict("this was terrible"))                # "negative"
```

## Part 3: Teach a model to predict from a spreadsheet

```python
import random
random.seed(0)
rows = ["age,income,city,label"]
for _ in range(400):
    age = random.randint(18, 70)
    income = random.randint(20000, 150000)
    city = random.choice(["NYC", "LA", "Chicago"])
    label = 1 if income > 70000 else 0
    rows.append(f"{age},{income},{city},{label}")
open("customers.csv", "w").write("\n".join(rows))
```

```python
model = tl.train("customers.csv")
print(model.predict({"age": 35, "income": 95000, "city": "NYC"}))  # 1
```

Tensorless found the `label` column, saw it looked categorical, and
trained a classifier. If you rename `label` to `price` and fill it with
dollar amounts instead of 0/1, Tensorless will detect **regression**
instead — no code changes needed, it reads the shape of your data.

## Part 4: The Smart Auto Check in action

Run the exact same training call again:

```python
model = tl.train("customers.csv")
```

This returns instantly — Tensorless recognized the dataset hadn't
changed and the model was already trained, so it just handed back the
existing `model.tl`.

Now change the data and run it again:

```python
open("customers.csv", "a").write("\n40,60000,LA,0")
model = tl.train("customers.csv")  # retrains, because the data changed
```

## What's next

- [Training](training.md) for the full picture of what `tl.train()`
  supports
- [Configuration](configuration.md) to see every parameter you can
  override once you outgrow the defaults
- [Examples](examples.md) for more worked examples
