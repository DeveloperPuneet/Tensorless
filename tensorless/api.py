"""Public Tensorless API.

    import tensorless as tl

    tl.train("./data")            # full auto
    tl.train("./data", d_model=512, layers=6, batch_size=32)  # override anything
    tl.inspect("./data")
    model = tl.load("model.tl")
    tl.run("model.tl")
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import os
from typing import Any, Optional

from .config import TrainConfig
from .errors import ConfigError, DataError, ModelError
from .data.loader import load_dataset
from .data.fingerprint import fingerprint_path
from .data.inspector import inspect_path, InspectionReport
from .auto.config import resolve_config
from .checkpoint.manager import CheckpointManager
from .training.trainer import run_training
from .serialization.tl_format import save_tl, load_tl
from .runtime import LoadedModel, load_model
from ._version import __version__ as _tl_version, TL_FORMAT_VERSION as _tl_format_version

_KNOWN_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def _build_train_config(**kwargs: Any) -> TrainConfig:
    unknown = set(kwargs) - _KNOWN_FIELDS
    if unknown:
        raise ConfigError(
            f"Unknown train() argument(s): {sorted(unknown)}. "
            f"Valid arguments: {sorted(_KNOWN_FIELDS)}"
        )
    return TrainConfig(**kwargs)


def inspect(path: str) -> InspectionReport:
    """Inspect a dataset: detected task, size, problems, recommendations.

    Does not train anything. Prints a human-readable report and also
    returns a structured `InspectionReport` object.
    """
    report = inspect_path(path)
    print(report)
    return report


def train(path: str, **kwargs: Any) -> LoadedModel:
    """Train a model on the dataset at `path`, fully automatically by
    default. Any field of `TrainConfig` can be overridden via keyword
    argument, e.g. `tl.train("./data", d_model=512, layers=6)`.

    Implements the "Smart Auto Check":
      - if an up-to-date trained model already exists for this exact
        dataset -> return it without retraining
      - if training was interrupted on this exact dataset -> resume
      - if the dataset changed -> retrain (or raise, if
        `ask_on_data_change=True`), unless `force=True` is passed
    """
    user_cfg = _build_train_config(**kwargs)
    out = user_cfg.out or "model.tl"
    checkpoint_dir = user_cfg.checkpoint_dir or (out + ".ckpt")
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    if not os.path.exists(path):
        raise DataError(f"Path '{path}' does not exist.")

    fingerprint = fingerprint_path(path)

    resume_state = None

    if not user_cfg.force:
        # 1. Is there already a complete, up-to-date .tl file?
        try:
            existing = load_tl(out)
        except Exception:
            existing = None
        if existing is not None:
            if existing.get("dataset_fingerprint") == fingerprint and existing.get("training_complete"):
                if user_cfg.verbose:
                    print(
                        f"[tensorless] '{out}' already exists and matches this "
                        f"dataset (fingerprint {fingerprint[:12]}...) -- using "
                        f"the existing model. Pass force=True to retrain."
                    )
                return LoadedModel(existing)

        # 2. Is there an interrupted / matching checkpoint to resume from?
        if checkpoint_mgr.exists() and user_cfg.resume is not False:
            ckpt = checkpoint_mgr.load()
            if ckpt.get("dataset_fingerprint") == fingerprint:
                if ckpt.get("training_complete"):
                    # Training finished but the final .tl wasn't written
                    # (e.g. process died right after the last checkpoint).
                    # No need to retrain -- just package the .tl file.
                    return _finalize_from_checkpoint(ckpt, out, user_cfg.verbose)
                resume_state = ckpt
                if user_cfg.verbose:
                    print(
                        f"[tensorless] found an interrupted checkpoint for this "
                        f"exact dataset -- resuming training."
                    )
            else:
                if user_cfg.ask_on_data_change:
                    raise ConfigError(
                        "The dataset has changed since the existing checkpoint "
                        "was created. Pass force=True to retrain from scratch, "
                        "or ask_on_data_change=False to retrain automatically."
                    )
                if user_cfg.verbose:
                    print(
                        "[tensorless] dataset has changed since the last "
                        "checkpoint -- retraining from scratch."
                    )
                checkpoint_mgr.clear()
        elif checkpoint_mgr.exists():
            if user_cfg.verbose:
                print("[tensorless] resume=False -- ignoring the existing checkpoint.")
            checkpoint_mgr.clear()
    else:
        checkpoint_mgr.clear()

    ds = load_dataset(path)
    resolved = resolve_config(ds, user_cfg)
    cfg = resolved.to_dict()

    if resume_state is not None:
        # Resumed runs must keep the exact architecture/config used
        # originally, regardless of any new overrides, so the checkpoint
        # can actually be loaded.
        cfg = resume_state["config"]

    result = run_training(
        ds=ds,
        cfg=cfg,
        checkpoint_mgr=checkpoint_mgr,
        dataset_fingerprint=fingerprint,
        resume_state=resume_state,
        log_fn=print if cfg.get("verbose", True) else (lambda *a, **k: None),
    )

    payload = {
        "tl_format_version": _tl_format_version,
        "tensorless_version": _tl_version,
        "task": cfg["task"],
        "model_type": cfg["model_type"],
        "config": cfg,
        "meta": result["meta"],
        "model_state_dict": result["model_state_dict"],
        "tokenizer_state": result["tokenizer"].state_dict() if result["tokenizer"] else None,
        "preprocessor_state": result["preprocessor"].state_dict() if result["preprocessor"] else None,
        "dataset_fingerprint": fingerprint,
        "training_complete": True,
        "metrics": result["metrics"],
    }
    save_tl(out, payload)
    if cfg.get("verbose", True):
        print(f"[tensorless] saved trained model to '{out}'")

    return LoadedModel(payload)


def pretrain(
    out: str = "english_pretrained.tl", language: str = "english", **kwargs: Any
) -> LoadedModel:
    """Pretrain a small language model on the built-in starter corpus."""
    if language.lower() != "english":
        raise ValueError("The built-in pretraining corpus currently supports only 'english'.")
    corpus = importlib.resources.files("tensorless.data").joinpath("english_grammar.txt")
    options = {"task": "text-generation", "out": out, **kwargs}
    return train(str(corpus), **options)


def _finalize_from_checkpoint(ckpt: dict, out: str, verbose: bool) -> LoadedModel:
    model_state = ckpt.get("best_model_state_dict") or ckpt["model_state_dict"]
    payload = {
        "tl_format_version": _tl_format_version,
        "tensorless_version": _tl_version,
        "task": ckpt["config"]["task"],
        "model_type": ckpt["config"]["model_type"],
        "config": ckpt["config"],
        "meta": ckpt["meta"],
        "model_state_dict": model_state,
        "tokenizer_state": ckpt.get("tokenizer_state"),
        "preprocessor_state": ckpt.get("preprocessor_state"),
        "dataset_fingerprint": ckpt["dataset_fingerprint"],
        "training_complete": True,
        "metrics": {},
    }
    save_tl(out, payload)
    if verbose:
        print(f"[tensorless] finalized already-complete checkpoint into '{out}'")
    return LoadedModel(payload)


def load(path: str, device: Optional[str] = None) -> LoadedModel:
    """Load a trained `.tl` model for inference."""
    return load_model(path, device=device)


def run(path: str, prompt: Optional[str] = None) -> Any:
    """Run a trained `.tl` model.

    For text-generation models with no prompt given, starts an
    interactive terminal chat. Otherwise runs one generation/prediction
    and returns/prints the result.
    """
    model = load_model(path)
    if model.task == "text-generation":
        if prompt is None:
            model.chat()
            return None
        result = model.generate(prompt)
        print(result)
        return result
    elif model.task == "text-classification":
        if prompt is None:
            print(
                f"Loaded a '{model.task}' model. Use tl.load('{path}').predict(text) "
                f"to classify text, or pass prompt=... / --prompt on the CLI."
            )
            return None
        result = model.predict(prompt)
        print(result)
        return result
    else:
        # tabular classification/regression: needs a structured record, not
        # free text, so the CLI --prompt shortcut doesn't apply here.
        if prompt is not None:
            raise ModelError(
                f"'{path}' is a tabular '{model.task}' model, which expects a "
                f"structured record (e.g. {{'age': 30, 'income': 90000}}), not "
                f"free text. Use tl.load('{path}').predict({{...}}) from Python "
                f"instead of --prompt on the CLI."
            )
        print(
            f"Loaded a tabular '{model.task}' model. Use "
            f"tl.load('{path}').predict({{...}}) to run predictions."
        )
        return None
