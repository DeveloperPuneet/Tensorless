import tensorless as tl
from tensorless.data.tabular import TabularPreprocessor
from tensorless.training.early_stopping import EarlyStopping

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


def test_datetime_columns_are_numeric_features():
    records = [
        {"created": "2024-01-01T00:00:00Z", "label": "old"},
        {"created": "2024-01-02T00:00:00Z", "label": "new"},
    ]
    prep = TabularPreprocessor().fit(records, ["created", "label"], "label", "classification")
    assert prep.numeric_columns == ["created"]
    transformed = prep.transform(records)
    assert transformed["numeric"].shape == (2, 1)
    assert transformed["numeric"][0, 0] < transformed["numeric"][1, 0]


def test_high_cardinality_categories_are_bounded():
    records = [{"user": f"user-{i}", "label": i % 2} for i in range(1100)]
    prep = TabularPreprocessor().fit(records, ["user", "label"], "label", "classification")
    assert len(prep.column_stats["user"].vocab) <= 1002
    transformed = prep.transform([{"user": "new-user"}], with_target=False)
    assert transformed["categorical"][0, 0].item() == 1


def test_regression_scaling_resists_extreme_outlier():
    records = [{"feature": i, "target": i} for i in range(10)]
    records.append({"feature": 10, "target": 1_000_000})
    prep = TabularPreprocessor().fit(records, ["feature", "target"], "target", "regression")
    assert prep.target_mean < 10
    assert prep.target_std < 20


def test_early_stopping_stops_after_three_bad_epochs():
    stopping = EarlyStopping(patience=3)
    assert stopping.step(1.0) is True
    assert stopping.step(1.1) is False
    assert stopping.step(1.1) is False
    assert stopping.step(1.1) is False
    assert stopping.should_stop is True
