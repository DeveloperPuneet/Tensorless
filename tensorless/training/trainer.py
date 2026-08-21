"""The training loop.

`run_training` is the single entry point that: prepares data, builds (or
resumes) the model + optimizer, trains with early stopping, checkpoints
periodically so interrupted runs can resume, and returns everything
needed to write the final `.tl` file.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.loader import Dataset
from ..devices.device import get_torch_device
from ..checkpoint.manager import CheckpointManager
from ..models.registry import build_model
from ..tokenization.char_tokenizer import CharTokenizer
from ..data.tabular import TabularPreprocessor
from .early_stopping import EarlyStopping
from . import data_prep as dp


def _build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 1.0 - progress)


def _compute_loss(task: str, model_type: str, model: nn.Module, batch, device, pad_id: int) -> torch.Tensor:
    if model_type == "transformer" and task == "text-generation":
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id
        )
        return loss
    elif model_type == "transformer" and task == "text-classification":
        input_ids, attn_mask, labels = batch
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        logits = model(input_ids, attention_mask=attn_mask)
        return F.cross_entropy(logits, labels)
    elif model_type == "mlp" and task == "classification":
        numeric, categorical, target = batch
        numeric, categorical, target = numeric.to(device), categorical.to(device), target.to(device)
        logits = model(numeric, categorical)
        return F.cross_entropy(logits, target)
    elif model_type == "mlp" and task == "regression":
        numeric, categorical, target = batch
        numeric, categorical, target = numeric.to(device), categorical.to(device), target.to(device)
        pred = model(numeric, categorical)
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unsupported task/model_type combination: {task}/{model_type}")


def run_training(
    ds: Dataset,
    cfg: Dict[str, Any],
    checkpoint_mgr: CheckpointManager,
    dataset_fingerprint: str,
    resume_state: Optional[Dict[str, Any]] = None,
    log_fn=print,
) -> Dict[str, Any]:
    task = cfg["task"]
    model_type = cfg["model_type"]
    torch.manual_seed(cfg["seed"])

    device = get_torch_device(cfg["device"])
    if cfg["verbose"]:
        log_fn(f"[tensorless] task={task} model={model_type} device={cfg['device']} precision={cfg['precision']}")

    # ---- data prep (resume tokenizer/preprocessor if available) ----
    tokenizer = None
    preprocessor = None
    if resume_state is not None:
        if resume_state.get("tokenizer_state") is not None:
            tokenizer = CharTokenizer.from_state_dict(resume_state["tokenizer_state"])
        if resume_state.get("preprocessor_state") is not None:
            preprocessor = TabularPreprocessor.from_state_dict(resume_state["preprocessor_state"])

    if task == "text-generation":
        prepared = dp.prepare_text_generation(ds, cfg, tokenizer=tokenizer)
    elif task == "text-classification":
        classes = resume_state["meta"]["classes"] if resume_state else None
        prepared = dp.prepare_text_classification(ds, cfg, tokenizer=tokenizer, classes=classes)
    elif task in ("classification", "regression"):
        prepared = dp.prepare_tabular(ds, cfg, task=task, preprocessor=preprocessor)
    else:
        raise ValueError(f"Unsupported task '{task}'")

    # ---- model / optimizer / scheduler ----
    model = build_model(task, model_type, cfg, prepared.meta).to(device)
    optimizer = _build_optimizer(model, cfg)

    steps_per_epoch = max(1, len(prepared.train_loader))
    total_steps = cfg.get("max_steps") or steps_per_epoch * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, cfg["warmup_steps"], total_steps)
    )

    early_stopper = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])
    start_epoch = 0
    global_step = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        early_stopper.best = resume_state.get("early_stopping_best", float("inf"))
        early_stopper.num_bad_checks = resume_state.get("early_stopping_bad_checks", 0)
        if cfg["verbose"]:
            log_fn(f"[tensorless] resuming from checkpoint: epoch={start_epoch}, step={global_step}")

    pad_id = prepared.meta.get("pad_id", 0)

    def _checkpoint(epoch: int, training_complete: bool) -> None:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping_best": early_stopper.best,
            "early_stopping_bad_checks": early_stopper.num_bad_checks,
            "config": cfg,
            "meta": prepared.meta,
            "tokenizer_state": prepared.tokenizer.state_dict() if prepared.tokenizer else None,
            "preprocessor_state": prepared.preprocessor.state_dict() if prepared.preprocessor else None,
            "dataset_fingerprint": dataset_fingerprint,
            "training_complete": training_complete,
        }
        checkpoint_mgr.save(state)

    # ---- training loop ----
    model.train()
    stop = False
    t0 = time.time()
    last_val_loss = None
    last_train_loss = None

    for epoch in range(start_epoch, cfg["epochs"]):
        for batch in prepared.train_loader:
            loss = _compute_loss(task, model_type, model, batch, device, pad_id)
            last_train_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % cfg["checkpoint_every"] == 0:
                _checkpoint(epoch, training_complete=False)

            if cfg.get("max_steps") and global_step >= cfg["max_steps"]:
                stop = True
                break
        if stop:
            break

        # ---- validation / early stopping ----
        if prepared.val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in prepared.val_loader:
                    losses.append(_compute_loss(task, model_type, model, batch, device, pad_id).item())
            val_loss = sum(losses) / max(1, len(losses))
            last_val_loss = val_loss
            model.train()
            is_best = early_stopper.step(val_loss, state=None)
            if cfg["verbose"]:
                log_fn(
                    f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} "
                    f"train_loss={last_train_loss:.4f} val_loss={val_loss:.4f}"
                    f"{' (best)' if is_best else ''}"
                )
            if early_stopper.should_stop:
                if cfg["verbose"]:
                    log_fn(f"[tensorless] early stopping at epoch {epoch + 1} (no improvement)")
                _checkpoint(epoch, training_complete=True)
                break
        else:
            if cfg["verbose"]:
                log_fn(f"[tensorless] epoch {epoch + 1}/{cfg['epochs']} train_loss={last_train_loss:.4f}")

        _checkpoint(epoch + 1, training_complete=(epoch + 1 >= cfg["epochs"]))

    elapsed = time.time() - t0
    if cfg["verbose"]:
        log_fn(f"[tensorless] training finished in {elapsed:.1f}s ({global_step} steps)")

    metrics = {
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "global_step": global_step,
        "elapsed_seconds": elapsed,
    }

    return {
        "model": model,
        "model_state_dict": model.state_dict(),
        "meta": prepared.meta,
        "tokenizer": prepared.tokenizer,
        "preprocessor": prepared.preprocessor,
        "metrics": metrics,
        "device": device,
    }
