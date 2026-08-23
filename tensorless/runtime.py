"""Runtime inference wrapper.

`tl.load("model.tl")` returns a `LoadedModel`, which knows how to rebuild
the exact architecture used at training time, load the weights, and
expose a task-appropriate prediction API:

  - text-generation      -> `.generate(prompt)` and `.chat()`
  - text-classification  -> `.predict(text)`
  - classification/regression (tabular) -> `.predict(record_or_records)`
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from .models.registry import build_model
from .tokenization import tokenizer_from_state_dict
from .data.tabular import TabularPreprocessor
from .devices.device import get_device
from .errors import ModelError
from .serialization.tl_format import load_tl


class LoadedModel:
    def __init__(self, payload: Dict[str, Any], device: str = None):
        self.payload = payload
        self.task: str = payload["task"]
        self.model_type: str = payload["model_type"]
        self.config: Dict[str, Any] = payload["config"]
        self.meta: Dict[str, Any] = payload["meta"]
        self.metrics: Dict[str, Any] = payload.get("metrics", {})
        self.dataset_fingerprint: str = payload.get("dataset_fingerprint")

        self.tokenizer = None
        if payload.get("tokenizer_state") is not None:
            self.tokenizer = tokenizer_from_state_dict(payload["tokenizer_state"])

        self.preprocessor = None
        if payload.get("preprocessor_state") is not None:
            self.preprocessor = TabularPreprocessor.from_state_dict(payload["preprocessor_state"])

        device_name = device or self.config.get("device", "cpu")
        self.device = get_device(device_name)

        backend = "numpy"
        if self.device in ("cuda", "tpu"):
            backend = "jax"
        elif self.device == "mps":
            backend = "mlx"
        self.model = build_model(self.task, self.model_type, self.config, self.meta, backend=backend)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str = "", max_new_tokens: int = 200, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
        if self.task != "text-generation":
            raise ModelError(f"generate() is only available for text-generation models, not '{self.task}'.")
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)[:-1]  # drop trailing eos
        if not ids:
            ids = [self.tokenizer.bos_id]
        input_ids = np.asarray([ids], dtype=np.int64)
        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=self.tokenizer.eos_id,
        )
        return self.tokenizer.decode(out[0].tolist())

    def chat(self) -> None:
        """Interactive terminal chat loop for text-generation models."""
        if self.task != "text-generation":
            raise ModelError(f"chat() is only available for text-generation models, not '{self.task}'.")
        print("Tensorless interactive chat. Type 'exit' or Ctrl+C to quit.")
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() in ("exit", "quit"):
                break
            reply = self.generate(prompt, max_new_tokens=200)
            print(reply)

    # ------------------------------------------------------------------
    # Text classification
    # ------------------------------------------------------------------
    def _predict_text_classification(self, texts: List[str]) -> List[str]:
        block_size = self.config["max_seq_len"]
        batch_ids, batch_mask = [], []
        for t in texts:
            ids = self.tokenizer.encode(t, add_special_tokens=True)[:block_size]
            mask = [1] * len(ids)
            if len(ids) < block_size:
                pad_len = block_size - len(ids)
                ids = ids + [self.tokenizer.pad_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)
        input_ids = np.asarray(batch_ids, dtype=np.int64)
        attn_mask = np.asarray(batch_mask, dtype=np.float32)
        logits = self.model.forward(input_ids, attention_mask=attn_mask)
        pred_idx = logits.argmax(axis=-1).tolist()
        classes = self.meta["classes"]
        return [classes[i] for i in pred_idx]

    # ------------------------------------------------------------------
    # Tabular classification / regression
    # ------------------------------------------------------------------
    def _predict_tabular(self, records: List[Dict[str, Any]]) -> List[Any]:
        transformed = self.preprocessor.transform(records, with_target=False)
        numeric = transformed["numeric"]
        categorical = transformed["categorical"]
        out = self.model.forward(numeric, categorical)
        if self.task == "classification":
            pred_idx = out.argmax(axis=-1)
            return self.preprocessor.inverse_target(pred_idx)
        else:
            return self.preprocessor.inverse_target(out)

    # ------------------------------------------------------------------
    # Unified predict()
    # ------------------------------------------------------------------
    def predict(self, x: Union[str, Dict[str, Any], List[Any]]) -> Any:
        single = not isinstance(x, list)
        items = [x] if single else x

        if self.task == "text-classification":
            preds = self._predict_text_classification(items)
        elif self.task in ("classification", "regression"):
            preds = self._predict_tabular(items)
        elif self.task == "text-generation":
            preds = [self.generate(prompt=str(i)) for i in items]
        else:
            raise ModelError(f"predict() not supported for task '{self.task}'.")

        return preds[0] if single else preds

    def info(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model_type": self.model_type,
            "tensorless_version": self.payload.get("tensorless_version"),
            "tl_format_version": self.payload.get("tl_format_version"),
            "config": self.config,
            "metrics": self.metrics,
            "training_complete": self.payload.get("training_complete"),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
        }


def load_model(path: str, device: str = None) -> LoadedModel:
    payload = load_tl(path)
    return LoadedModel(payload, device=device)
