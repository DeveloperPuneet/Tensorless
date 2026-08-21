"""Example: train a tiny sentiment classifier from labeled text folders.

Run:
    python examples/text_classification_example.py
"""

import os
import random

import tensorless as tl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "reviews")


def make_data():
    random.seed(0)
    pos_words = ["great", "amazing", "wonderful", "excellent", "fantastic"]
    neg_words = ["terrible", "awful", "horrible", "bad", "disappointing"]
    os.makedirs(os.path.join(DATA_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "negative"), exist_ok=True)
    for i in range(60):
        w = random.choice(pos_words)
        with open(os.path.join(DATA_DIR, "positive", f"{i}.txt"), "w") as f:
            f.write(f"this movie was {w} and I really enjoyed watching it")
    for i in range(60):
        w = random.choice(neg_words)
        with open(os.path.join(DATA_DIR, "negative", f"{i}.txt"), "w") as f:
            f.write(f"this movie was {w} and I really did not enjoy it")


def main():
    if not os.path.isdir(DATA_DIR):
        make_data()

    tl.inspect(DATA_DIR)

    model = tl.train(
        DATA_DIR,
        out=os.path.join(os.path.dirname(DATA_DIR), "sentiment_example.tl"),
        epochs=15,
        d_model=32,
        layers=2,
        heads=2,
        batch_size=16,
        max_seq_len=64,
    )

    print(model.predict("this movie was wonderful and I really enjoyed watching it"))
    print(model.predict("this movie was terrible and I really did not enjoy it"))


if __name__ == "__main__":
    main()
