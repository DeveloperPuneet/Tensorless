"""Example: train a tiny tabular classifier from a CSV file.

Run:
    python examples/tabular_classification_example.py
"""

import os
import random

import tensorless as tl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "customers.csv")


def make_data():
    random.seed(0)
    lines = ["age,income,city,label"]
    for _ in range(400):
        age = random.randint(18, 70)
        income = random.randint(20000, 150000)
        city = random.choice(["NYC", "LA", "Chicago"])
        label = 1 if income > 70000 else 0
        lines.append(f"{age},{income},{city},{label}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CSV_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    if not os.path.exists(CSV_PATH):
        make_data()

    tl.inspect(CSV_PATH)

    model = tl.train(
        CSV_PATH,
        out=os.path.join(DATA_DIR, "churn_example.tl"),
        epochs=15,
        d_model=32,
        layers=2,
        batch_size=32,
    )

    print(model.predict({"age": 35, "income": 95000, "city": "NYC"}))
    print(model.predict({"age": 25, "income": 30000, "city": "LA"}))


if __name__ == "__main__":
    main()
