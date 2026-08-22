import os

import tensorless as tl
from tensorless.devices.device import auto_select_device

from .conftest import TINY_TEXT_KWARGS


def test_full_end_to_end_pipeline(text_corpus, workdir):
    """dataset -> auto detection -> auto config -> training -> checkpoint
    -> .tl file -> reload -> inference, all in one flow."""
    report = tl.inspect(text_corpus)
    assert report.task == "text-generation"

    model = tl.train(text_corpus, out="e2e.tl", **TINY_TEXT_KWARGS)
    assert os.path.isfile("e2e.tl")
    assert os.path.isdir("e2e.tl.ckpt")

    reloaded = tl.load("e2e.tl")
    output = reloaded.generate("the quick", max_new_tokens=20)
    assert isinstance(output, str) and len(output) > 0

    info = reloaded.info()
    assert info["training_complete"] is True
    assert info["n_parameters"] > 0


def test_device_auto_selection_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr("tensorless.devices.device._cuda_available", lambda: False)
    monkeypatch.setattr("tensorless.devices.device._tpu_available", lambda: False)
    monkeypatch.setattr("tensorless.devices.device._mps_available", lambda: False)
    device, precision = auto_select_device(None, None)
    assert device == "cpu"
    assert precision == "fp32"


def test_device_respects_explicit_cpu_request(text_corpus, workdir):
    model = tl.train(text_corpus, out="model.tl", device="cpu", **TINY_TEXT_KWARGS)
    assert model.config["device"] == "cpu"


def test_cuda_training_if_available(text_corpus, workdir):
    import pytest

    pytest.skip("The native engine currently provides CPU kernels only")
