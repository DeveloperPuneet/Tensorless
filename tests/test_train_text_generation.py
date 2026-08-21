import os

import tensorless as tl
from tensorless.tokenization.bpe_tokenizer import BPETokenizer
from tensorless.serialization.tl_format import load_tl
from tensorless.auto.config import resolve_config
from tensorless.config import TrainConfig
from tensorless.data.loader import Dataset
from tensorless.training.data_prep import prepare_text_generation

from .conftest import TINY_TEXT_KWARGS


def test_train_text_generation_creates_tl_file(text_corpus, workdir):
    model = tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    assert os.path.isfile("model.tl")
    assert model.task == "text-generation"
    assert isinstance(model.tokenizer, BPETokenizer)
    assert model.info()["training_complete"] is True


def test_generate_after_reload(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    reloaded = tl.load("model.tl")
    text = reloaded.generate("the quick", max_new_tokens=20)
    assert isinstance(text, str)
    assert len(text) > 0


def test_bpe_tokenizer_trains_and_reloads(text_corpus, workdir):
    model = tl.train(
        text_corpus,
        out="bpe.tl",
        tokenizer="bpe",
        bpe_vocab_size=64,
        **TINY_TEXT_KWARGS,
    )
    assert isinstance(model.tokenizer, BPETokenizer)
    assert model.tokenizer.decode(model.tokenizer.encode("the quick")) == "the quick"

    reloaded = tl.load("bpe.tl")
    assert isinstance(reloaded.tokenizer, BPETokenizer)
    assert reloaded.tokenizer.decode(reloaded.tokenizer.encode("the quick")) == "the quick"


def test_bpe_skips_duplicate_merges_and_preserves_text():
    tokenizer = BPETokenizer.build(["abab abac", "abab abac"], vocab_size=64)
    text = "abab abac"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_builtin_english_pretraining(workdir):
    model = tl.pretrain(
        out="english.tl", epochs=1, max_steps=1, max_seq_len=32,
        d_model=16, layers=1, heads=2, batch_size=2, checkpoint_every=1,
        verbose=False,
    )
    assert model.task == "text-generation"
    assert model.tokenizer is not None


def test_long_text_generation_is_streamed_in_batches():
    ds = Dataset(kind="text", source="memory", texts=["the quick brown fox " * 2000])
    cfg = resolve_config(ds, TrainConfig(max_seq_len=32, tokenizer="char"))
    prepared = prepare_text_generation(ds, cfg.to_dict())
    batch = next(iter(prepared.train_loader))
    assert batch[0].shape == (cfg.batch_size, 32)
    assert len(prepared.train_loader.dataset) > cfg.batch_size


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
