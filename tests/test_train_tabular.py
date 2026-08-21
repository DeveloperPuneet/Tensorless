import tensorless as tl

from .conftest import TINY_TABULAR_KWARGS


def test_train_tabular_classification(tabular_classification_csv, workdir):
    model = tl.train(tabular_classification_csv, out="cls.tl", **TINY_TABULAR_KWARGS)
    assert model.task == "classification"
    pred = model.predict({"age": 30, "income": 90000, "city": "NYC"})
    assert pred in ("0", "1")


def test_tabular_classification_learns_signal(tabular_classification_csv, workdir):
    model = tl.train(
        tabular_classification_csv, out="cls.tl", epochs=15, d_model=32, layers=2, batch_size=32
    )
    high_income_pred = model.predict({"age": 30, "income": 140000, "city": "NYC"})
    low_income_pred = model.predict({"age": 30, "income": 21000, "city": "NYC"})
    assert high_income_pred == "1"
    assert low_income_pred == "0"


def test_train_tabular_regression(tabular_regression_csv, workdir):
    model = tl.train(tabular_regression_csv, out="reg.tl", **TINY_TABULAR_KWARGS)
    assert model.task == "regression"
    pred = model.predict({"sqft": 2000, "bedrooms": 3, "city": "NYC"})
    assert isinstance(pred, float)


def test_tabular_regression_reasonable_magnitude(tabular_regression_csv, workdir):
    model = tl.train(tabular_regression_csv, out="reg.tl", epochs=20, d_model=32, layers=2, batch_size=32)
    pred = model.predict({"sqft": 2000, "bedrooms": 3, "city": "NYC"})
    # true relationship: price ~= sqft*150 + bedrooms*10000 = 330000
    assert 150000 < pred < 500000


def test_tabular_batch_predict(tabular_classification_csv, workdir):
    model = tl.train(tabular_classification_csv, out="cls.tl", **TINY_TABULAR_KWARGS)
    preds = model.predict(
        [
            {"age": 30, "income": 90000, "city": "NYC"},
            {"age": 25, "income": 30000, "city": "LA"},
        ]
    )
    assert isinstance(preds, list)
    assert len(preds) == 2
