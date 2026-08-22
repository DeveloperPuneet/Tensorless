"""NumPy training loop with resumable checkpoints and early stopping."""

from __future__ import annotations

import time
import sys
from typing import Any, Dict, Optional
import numpy as np

from ..data.loader import Dataset
from ..devices.device import get_device
from ..checkpoint.manager import CheckpointManager
from ..models.registry import build_model
from ..tokenization import tokenizer_from_state_dict
from ..data.tabular import TabularPreprocessor
from .early_stopping import EarlyStopping
from . import data_prep as dp
from ..engine import Adam, SGD, LambdaScheduler, softmax_cross_entropy


def _build_optimizer(model, cfg):
    name, lr, wd = cfg["optimizer"].lower(), cfg["learning_rate"], cfg["weight_decay"]
    if name == "adamw": return Adam(model.parameters(), lr, wd, decoupled=True)
    if name == "adam": return Adam(model.parameters(), lr, wd)
    if name == "sgd": return SGD(model.parameters(), lr, wd, momentum=.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def _compute_loss(task, model_type, model, batch, pad_id, backward=True):
    if model_type == "transformer" and task == "text-generation":
        return model.loss_and_backward(batch[0], batch[1]) if backward else softmax_cross_entropy(model.forward(batch[0]), batch[1], pad_id)[0]
    if model_type == "transformer" and task == "text-classification":
        if backward: return model.loss_and_backward(batch[0], batch[2], batch[1])
        return softmax_cross_entropy(model.forward(batch[0], batch[1]), batch[2])[0]
    numeric, categorical, target = batch
    if backward: return model.loss_and_backward(numeric, categorical, target)
    output = model.forward(numeric, categorical)
    if task == "regression": return float(np.mean((output - target) ** 2))
    return softmax_cross_entropy(output, target)[0]


def _show_progress(epoch, epochs, step, total_steps, loss):
    width = 24
    completed = min(width, int(width * step / max(1, total_steps)))
    bar = "=" * completed + ">" * (completed < width) + " " * max(0, width - completed - 1)
    print(
        f"\r[tensorless] epoch {epoch}/{epochs} [{bar}] "
        f"{step}/{total_steps} loss={loss:.4f}",
        end="",
        flush=True,
    )


def run_training(ds: Dataset, cfg: Dict[str, Any], checkpoint_mgr: CheckpointManager,
                 dataset_fingerprint: str, resume_state: Optional[Dict[str, Any]] = None,
                 pretrained_state: Optional[Dict[str, Any]] = None,
                 log_fn=print) -> Dict[str, Any]:
    np.random.seed(cfg["seed"])
    task, model_type = cfg["task"], cfg["model_type"]
    device = get_device(cfg["device"])
    if cfg["verbose"]: log_fn(f"[tensorless] task={task} model={model_type} device={cfg['device']} precision={cfg['precision']}")
    source_state = resume_state or pretrained_state
    tokenizer = tokenizer_from_state_dict(source_state["tokenizer_state"]) if source_state and source_state.get("tokenizer_state") else None
    preprocessor = TabularPreprocessor.from_state_dict(source_state["preprocessor_state"]) if source_state and source_state.get("preprocessor_state") else None
    if task == "text-generation": prepared = dp.prepare_text_generation(ds, cfg, tokenizer=tokenizer)
    elif task == "text-classification": prepared = dp.prepare_text_classification(ds, cfg, tokenizer=tokenizer, classes=resume_state["meta"]["classes"] if resume_state else None)
    elif task in ("classification", "regression"): prepared = dp.prepare_tabular(ds, cfg, task, preprocessor)
    else: raise ValueError(f"Unsupported task '{task}'")
    backend = "numpy"
    if model_type == "transformer" and task in ("text-generation", "text-classification"):
        if device in ("cuda", "tpu"):
            backend = "jax"
        elif device == "mps":
            backend = "mlx"
    model = build_model(task, model_type, cfg, prepared.meta, backend=backend)
    if pretrained_state is not None:
        try:
            model.load_state_dict(pretrained_state["model_state_dict"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Pretrained weights are incompatible with the target model. "
                "Keep the task, tokenizer, and architecture dimensions compatible."
            ) from exc
    optimizer = _build_optimizer(model, cfg)
    total_steps = cfg.get("max_steps") or max(1, len(prepared.train_loader)) * cfg["epochs"]
    scheduler = LambdaScheduler(optimizer, cfg["warmup_steps"], total_steps)
    early_stopper = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])
    start_epoch = global_step = 0
    best_model_state = None
    if resume_state:
        model.load_state_dict(resume_state["model_state_dict"]); optimizer.load_state_dict(resume_state["optimizer_state_dict"]); scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch, global_step = resume_state["epoch"], resume_state["global_step"]
        prepared.train_loader.epoch = resume_state.get("train_loader_epoch", start_epoch)
        early_stopper.best = resume_state.get("early_stopping_best", float("inf")); early_stopper.num_bad_checks = resume_state.get("early_stopping_bad_checks", 0)
        if resume_state.get("best_model_state_dict") is not None:
            best_model_state = resume_state["best_model_state_dict"]
    def checkpoint(epoch, complete):
        checkpoint_mgr.save({"epoch": epoch, "global_step": global_step, "train_loader_epoch": prepared.train_loader.epoch, "model_state_dict": model.state_dict(), "best_model_state_dict": best_model_state, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "early_stopping_best": early_stopper.best, "early_stopping_bad_checks": early_stopper.num_bad_checks, "config": cfg, "meta": prepared.meta, "tokenizer_state": prepared.tokenizer.state_dict() if prepared.tokenizer else None, "preprocessor_state": prepared.preprocessor.state_dict() if prepared.preprocessor else None, "dataset_fingerprint": dataset_fingerprint, "training_complete": complete})
    last_train_loss = last_val_loss = None
    t0 = time.time(); stop = False
    progress_enabled = cfg["verbose"] and sys.stdout.isatty()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_step = 0
        total_epoch_steps = len(prepared.train_loader)
        for batch in prepared.train_loader:
            epoch_step += 1
            optimizer.zero_grad(); last_train_loss = _compute_loss(task, model_type, model, batch, prepared.meta.get("pad_id", 0));
            if not np.isfinite(last_train_loss):
                raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}: {last_train_loss}")
            if cfg["grad_clip"]:
                norm = np.sqrt(sum(float(np.sum(p.grad * p.grad)) for p in model.parameters()))
                if not np.isfinite(norm):
                    raise FloatingPointError(f"Non-finite gradient norm at step {global_step + 1}: {norm}")
                if norm > cfg["grad_clip"]:
                    for p in model.parameters(): p.grad *= cfg["grad_clip"] / (norm + 1e-12)
            optimizer.step(); scheduler.step(); global_step += 1
            if progress_enabled:
                _show_progress(epoch + 1, cfg["epochs"], epoch_step, total_epoch_steps, last_train_loss)
            if global_step % cfg["checkpoint_every"] == 0: checkpoint(epoch, False)
            if cfg.get("max_steps") and global_step >= cfg["max_steps"]: stop = True; break
        if progress_enabled:
            print()
        if stop:
            break
        if prepared.val_loader:
            model.eval(); losses = [_compute_loss(task, model_type, model, batch, prepared.meta.get("pad_id", 0), False) for batch in prepared.val_loader]; last_val_loss = sum(losses) / max(1, len(losses))
            if not np.isfinite(last_val_loss):
                raise FloatingPointError(f"Non-finite validation loss after epoch {epoch + 1}: {last_val_loss}")
            if last_val_loss < early_stopper.best - early_stopper.min_delta:
                best_model_state = model.state_dict()
            early_stopper.step(last_val_loss)
            if cfg["verbose"]:
                log_fn(
                    f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} "
                    f"train_loss={last_train_loss:.4f} val_loss={last_val_loss:.4f}"
                )
            if early_stopper.should_stop: checkpoint(epoch, True); break
        elif cfg["verbose"]:
            log_fn(
                f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} "
                f"train_loss={last_train_loss:.4f}"
            )
        checkpoint(epoch + 1, epoch + 1 >= cfg["epochs"])
    elapsed = time.time() - t0
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return {"model": model, "model_state_dict": model.state_dict(), "meta": prepared.meta, "tokenizer": prepared.tokenizer, "preprocessor": prepared.preprocessor, "metrics": {"final_train_loss": last_train_loss, "final_val_loss": last_val_loss, "global_step": global_step, "elapsed_seconds": elapsed}, "device": device}
