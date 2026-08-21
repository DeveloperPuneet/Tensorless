import os

import tensorless as tl
from tensorless.serialization.tl_format import load_tl

from .conftest import TINY_TEXT_KWARGS


def test_train_text_generation_creates_tl_file(text_corpus, workdir):
    model = tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    assert os.path.isfile("model.tl")
    assert model.task == "text-generation"
    assert model.info()["training_complete"] is True


def test_generate_after_reload(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    reloaded = tl.load("model.tl")
    text = reloaded.generate("the quick", max_new_tokens=20)
    assert isinstance(text, str)
    assert len(text) > 0


def test_cpu_training_works(text_corpus, workdir):
    model = tl.train(text_corpus, out="model.tl", device="cpu", **TINY_TEXT_KWARGS)
    assert model.config["device"] == "cpu"


def test_unchanged_dataset_skips_retraining(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_1 = os.path.getmtime("model.tl")

    import time

    time.sleep(0.05)
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_2 = os.path.getmtime("model.tl")

    # File should not have been rewritten -- the existing model was reused.
    assert mtime_1 == mtime_2


def test_changed_dataset_triggers_retrain(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_1 = os.path.getmtime("model.tl")

    with open(text_corpus, "a") as f:
        f.write("\nsome brand new sentence that changes the fingerprint")

    import time

    time.sleep(0.05)
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_2 = os.path.getmtime("model.tl")
    assert mtime_2 > mtime_1


def test_force_retrains_even_if_unchanged(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    mtime_1 = os.path.getmtime("model.tl")

    import time

    time.sleep(0.05)
    tl.train(text_corpus, out="model.tl", force=True, **TINY_TEXT_KWARGS)
    mtime_2 = os.path.getmtime("model.tl")
    assert mtime_2 > mtime_1


def test_ask_on_data_change_raises(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    with open(text_corpus, "a") as f:
        f.write("\nchanged!")

    import pytest

    with pytest.raises(tl.ConfigError):
        tl.train(text_corpus, out="model.tl", ask_on_data_change=True)
