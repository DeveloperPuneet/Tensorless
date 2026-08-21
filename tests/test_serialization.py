import os

import tensorless as tl
from tensorless.serialization.tl_format import save_tl, load_tl
from tensorless.errors import SerializationError

from .conftest import TINY_TEXT_KWARGS


def test_tl_roundtrip(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    payload = load_tl("model.tl")
    assert payload["task"] == "text-generation"
    assert "model_state_dict" in payload
    assert payload["training_complete"] is True


def test_tl_missing_file_raises(workdir):
    import pytest

    with pytest.raises(SerializationError):
        load_tl("does_not_exist.tl")


def test_tl_corrupt_file_raises(workdir):
    import pytest

    path = "corrupt.tl"
    with open(path, "w") as f:
        f.write("this is not a valid tl file")
    with pytest.raises(SerializationError):
        load_tl(path)


def test_tl_missing_required_fields_raises(workdir):
    import pytest
    import torch

    path = "incomplete.tl"
    torch.save({"task": "text-generation"}, path)
    with pytest.raises(SerializationError):
        load_tl(path)


def test_model_is_single_portable_file(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    assert os.path.isfile("model.tl")
    # A single file -- not a directory -- contains everything needed.
    assert not os.path.isdir("model.tl")
