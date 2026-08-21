"""Example: train a tiny tabular regression model from a CSV file.

Run:
    python examples/tabular_regression_example.py
"""

import os
import random

import tensorless as tl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "houses.csv")


def make_data():
    random.seed(0)
    lines = ["sqft,bedrooms,city,price"]
    for _ in range(400):
        sqft = random.randint(500, 3500)
        bedrooms = random.randint(1, 5)
        city = random.choice(["NYC", "LA", "Chicago"])
        price = sqft * 150 + bedrooms * 10000 + random.randint(-5000, 5000)
        lines.append(f"{sqft},{bedrooms},{city},{price}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CSV_PATH, "w") as f:
        f.write("\n".join(lines))


def main():
    if not os.path.exists(CSV_PATH):
        make_data()

    tl.inspect(CSV_PATH)

    model = tl.train(
        CSV_PATH,
        out=os.path.join(DATA_DIR, "prices_example.tl"),
        epochs=20,
        d_model=32,
        layers=2,
        batch_size=32,
    )

    print(model.predict({"sqft": 2000, "bedrooms": 3, "city": "NYC"}))


if __name__ == "__main__":
    main()
