import os

import tensorless as tl
from tensorless.data.loader import load_dataset
from tensorless.data.fingerprint import fingerprint_path
from tensorless.auto.config import resolve_config
from tensorless.config import TrainConfig
from tensorless.checkpoint.manager import CheckpointManager
from tensorless.training.trainer import run_training

from .conftest import TINY_TEXT_KWARGS


def test_checkpoint_created_during_training(text_corpus, workdir):
    tl.train(text_corpus, out="model.tl", **TINY_TEXT_KWARGS)
    assert os.path.isdir("model.tl.ckpt")
    assert os.path.isfile(os.path.join("model.tl.ckpt", "checkpoint.pt"))


def test_interrupted_training_resumes(text_corpus, workdir):
    """Simulate a crash mid-training (an incomplete checkpoint on disk with
    no final .tl file) and verify that calling train() again resumes
    instead of starting over, and successfully completes.
    """
    ds = load_dataset(text_corpus)
    fp = fingerprint_path(text_corpus)
    tiny_kwargs = dict(TINY_TEXT_KWARGS)
    tiny_kwargs["checkpoint_every"] = 2
    user_cfg = TrainConfig(out="model.tl", max_steps=6, **tiny_kwargs)
    resolved = resolve_config(ds, user_cfg)
    cfg = resolved.to_dict()
    ckpt_mgr = CheckpointManager(cfg["checkpoint_dir"])

    run_training(ds=ds, cfg=cfg, checkpoint_mgr=ckpt_mgr, dataset_fingerprint=fp, log_fn=lambda *a, **k: None)

    ckpt_before = ckpt_mgr.load()
    assert ckpt_before["training_complete"] is False
    assert ckpt_before["global_step"] == 6
    assert not os.path.isfile("model.tl")

    model = tl.train(text_corpus, out="model.tl")
    assert model.info()["training_complete"] is True
    assert os.path.isfile("model.tl")


def test_resume_uses_original_config_not_new_overrides(text_corpus, workdir):
    ds = load_dataset(text_corpus)
    fp = fingerprint_path(text_corpus)
    user_cfg = TrainConfig(
        out="model.tl", max_steps=4, checkpoint_every=2, d_model=16, layers=1, heads=2,
        max_seq_len=32, batch_size=16, epochs=1,
    )
    resolved = resolve_config(ds, user_cfg)
    cfg = resolved.to_dict()
    ckpt_mgr = CheckpointManager(cfg["checkpoint_dir"])
    run_training(ds=ds, cfg=cfg, checkpoint_mgr=ckpt_mgr, dataset_fingerprint=fp, log_fn=lambda *a, **k: None)

    # Even though we pass a different d_model here, resume must ignore it
    # and reuse the checkpoint's original architecture (otherwise loading
    # the checkpoint's weights would fail with a shape mismatch).
    model = tl.train(text_corpus, out="model.tl", d_model=999)
    assert model.config["d_model"] == 16


def test_resume_false_starts_from_scratch(text_corpus, workdir):
    ds = load_dataset(text_corpus)
    fp = fingerprint_path(text_corpus)
    original = TrainConfig(
        out="model.tl", max_steps=4, checkpoint_every=2, d_model=16, layers=1,
        heads=2, max_seq_len=32, batch_size=16, epochs=1,
    )
    cfg = resolve_config(ds, original).to_dict()
    ckpt_mgr = CheckpointManager(cfg["checkpoint_dir"])
    run_training(ds=ds, cfg=cfg, checkpoint_mgr=ckpt_mgr, dataset_fingerprint=fp, log_fn=lambda *a, **k: None)

    model = tl.train(
        text_corpus, out="model.tl", resume=False, d_model=24, layers=1,
        heads=2, max_seq_len=32, batch_size=16, epochs=1, verbose=False,
    )
    assert model.config["d_model"] == 24
