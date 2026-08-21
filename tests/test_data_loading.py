import json
import os

import pytest

from tensorless.data.loader import load_dataset
from tensorless.errors import DataError


def test_load_txt_file(text_corpus):
    ds = load_dataset(text_corpus)
    assert ds.kind == "text"
    assert len(ds) == 1
    assert "quick brown fox" in ds.texts[0]


def test_load_text_classification_dir(text_classification_dir):
    ds = load_dataset(text_classification_dir)
    assert ds.kind == "text_labeled"
    assert len(ds) == 80
    assert set(ds.labels) == {"positive", "negative"}


def test_load_csv(tabular_classification_csv):
    ds = load_dataset(tabular_classification_csv)
    assert ds.kind == "tabular"
    assert len(ds) == 300
    # Column order must be preserved (not alphabetized) so "last column"
    # target detection works correctly.
    assert ds.columns == ["age", "income", "city", "label"]


def test_load_jsonl_text(workdir):
    path = workdir / "data.jsonl"
    with open(path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"example number {i}"}) + "\n")
    ds = load_dataset(str(path))
    assert ds.kind == "text"
    assert len(ds) == 5


def test_load_json_records(workdir):
    path = workdir / "data.json"
    records = [{"a": i, "b": i * 2, "label": i % 2} for i in range(10)]
    path.write_text(json.dumps(records))
    ds = load_dataset(str(path))
    assert ds.kind == "tabular"
    assert len(ds) == 10


def test_missing_path_raises(workdir):
    with pytest.raises(DataError):
        load_dataset(str(workdir / "nope"))


def test_empty_dir_raises(workdir):
    d = workdir / "empty"
    d.mkdir()
    with pytest.raises(DataError):
        load_dataset(str(d))


def test_unsupported_extension_raises(workdir):
    path = workdir / "file.xyz"
    path.write_text("hello")
    with pytest.raises(DataError):
        load_dataset(str(path))


def test_non_utf8_file_raises(workdir):
    path = workdir / "bad.txt"
    with open(path, "wb") as f:
        f.write(b"\xff\xfe\x00bad")
    with pytest.raises(DataError):
        load_dataset(str(path))
