from tensorless.data.loader import load_dataset
from tensorless.auto.detector import detect_task


def test_detect_text_generation(text_corpus):
    ds = load_dataset(text_corpus)
    assert detect_task(ds) == "text-generation"


def test_detect_text_classification(text_classification_dir):
    ds = load_dataset(text_classification_dir)
    assert detect_task(ds) == "text-classification"


def test_detect_tabular_classification(tabular_classification_csv):
    ds = load_dataset(tabular_classification_csv)
    assert detect_task(ds) == "classification"


def test_detect_tabular_regression(tabular_regression_csv):
    ds = load_dataset(tabular_regression_csv)
    assert detect_task(ds) == "regression"
