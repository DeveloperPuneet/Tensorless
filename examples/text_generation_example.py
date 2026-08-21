"""Example: train a tiny text-generation model and sample from it.

Run:
    python examples/text_generation_example.py
"""

import os
import random

import tensorless as tl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CORPUS_PATH = os.path.join(DATA_DIR, "tiny_corpus.txt")


def make_data():
    random.seed(0)
    lines = [
        f"the quick brown fox jumps over the lazy dog number {i}" for i in range(300)
    ]
    with open(CORPUS_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    if not os.path.exists(CORPUS_PATH):
        make_data()

    tl.inspect(CORPUS_PATH)

    model = tl.train(
        CORPUS_PATH,
        out=os.path.join(DATA_DIR, "text_gen_example.tl"),
        # epochs=5,
        # d_model=64,
        # layers=2,
        # heads=4,
        # batch_size=16,
        # max_seq_len=64,
    )

    print("\nGenerated text:")
    print(model.generate("the quick", max_new_tokens=60))


if __name__ == "__main__":
    main()
