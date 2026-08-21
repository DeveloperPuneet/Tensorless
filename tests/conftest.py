import os
import random
import shutil

import pytest


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Run each test inside its own temp directory (so model.tl / checkpoints
    from one test never leak into another)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def text_corpus(workdir):
    random.seed(0)
    lines = [f"the quick brown fox jumps over the lazy dog number {i}" for i in range(300)]
    path = workdir / "corpus.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture()
def text_classification_dir(workdir):
    random.seed(1)
    pos_words = ["great", "amazing", "fantastic", "love", "wonderful", "excellent"]
    neg_words = ["terrible", "awful", "bad", "hate", "horrible", "worst"]
    root = workdir / "textcls"
    (root / "positive").mkdir(parents=True)
    (root / "negative").mkdir(parents=True)
    for i in range(40):
        w = random.choice(pos_words)
        (root / "positive" / f"p{i}.txt").write_text(
            f"this movie was {w} and I really enjoyed watching it", encoding="utf-8"
        )
    for i in range(40):
        w = random.choice(neg_words)
        (root / "negative" / f"n{i}.txt").write_text(
            f"this movie was {w} and I really did not enjoy it", encoding="utf-8"
        )
    return str(root)


@pytest.fixture()
def tabular_classification_csv(workdir):
    random.seed(2)
    lines = ["age,income,city,label"]
    for i in range(300):
        age = random.randint(18, 70)
        income = random.randint(20000, 150000)
        city = random.choice(["NYC", "LA", "Chicago"])
        label = 1 if income > 70000 else 0
        lines.append(f"{age},{income},{city},{label}")
    path = workdir / "tabular_cls.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture()
def tabular_regression_csv(workdir):
    random.seed(3)
    lines = ["sqft,bedrooms,city,price"]
    for i in range(300):
        sqft = random.randint(500, 3500)
        bedrooms = random.randint(1, 5)
        city = random.choice(["NYC", "LA", "Chicago"])
        price = sqft * 150 + bedrooms * 10000 + random.randint(-5000, 5000)
        lines.append(f"{sqft},{bedrooms},{city},{price}")
    path = workdir / "tabular_reg.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


TINY_TEXT_KWARGS = dict(
    epochs=1, max_seq_len=32, d_model=16, layers=1, heads=2, batch_size=16, checkpoint_every=5
)
TINY_TABULAR_KWARGS = dict(epochs=2, d_model=16, layers=1, batch_size=32)
