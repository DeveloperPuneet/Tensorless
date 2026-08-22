"""Tabular preprocessing.

Fits simple, fully-reversible preprocessing on tabular records:
    - numeric and datetime columns -> robustly scaled numeric features,
        with missing values imputed with the training-set median
    - categorical columns -> bounded frequency-ranked vocabulary (+ <unk>/<missing>),
        fed into per-column embeddings by the MLP model

The fitted state is small and JSON-serializable, so it can be embedded
directly inside a `.tl` file and reproduced exactly at inference time.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_MISSING = "<missing>"
_UNK = "<unk>"
_MAX_CATEGORIES = 1000


def _try_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _try_datetime(v: Any) -> Optional[float]:
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        value = v.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _robust_stats(values: List[float]) -> Tuple[float, float]:
    ordered = sorted(values)
    middle = ordered[len(ordered) // 2]
    if len(ordered) % 2 == 0:
        middle = (ordered[len(ordered) // 2 - 1] + middle) / 2
    deviations = sorted(abs(value - middle) for value in values)
    mad = deviations[len(deviations) // 2]
    if len(deviations) % 2 == 0:
        mad = (deviations[len(deviations) // 2 - 1] + mad) / 2
    scale = max(mad * 1.4826, 1e-6)
    if mad == 0.0 and len(values) > 1:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = max(variance ** 0.5, 1e-6)
    return middle, scale


@dataclass
class ColumnStats:
    kind: str  # "numeric" | "datetime" | "categorical"
    mean: float = 0.0
    std: float = 1.0
    vocab: List[str] = field(default_factory=list)


@dataclass
class TabularPreprocessor:
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task: str = "classification"
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)  # for classification targets
    target_mean: float = 0.0
    target_std: float = 1.0

    @property
    def numeric_columns(self) -> List[str]:
        return [c for c in self.feature_columns if self.column_stats[c].kind in ("numeric", "datetime")]

    @property
    def categorical_columns(self) -> List[str]:
        return [c for c in self.feature_columns if self.column_stats[c].kind == "categorical"]

    def fit(
        self,
        records: List[Dict[str, Any]],
        columns: List[str],
        target_column: str,
        task: str,
    ) -> "TabularPreprocessor":
        self.target_column = target_column
        self.task = task
        self.feature_columns = [c for c in columns if c != target_column]

        for col in self.feature_columns:
            values = [r.get(col) for r in records]
            numeric_vals = [_try_float(v) for v in values]
            n_present = sum(1 for v in values if v not in (None, ""))
            n_numeric = sum(1 for v in numeric_vals if v is not None)
            if n_present > 0 and n_numeric / n_present > 0.95:
                nums = [v for v in numeric_vals if v is not None]
                mean, std = _robust_stats(nums)
                self.column_stats[col] = ColumnStats(kind="numeric", mean=mean, std=std)
            else:
                datetime_vals = [_try_datetime(v) for v in values]
                n_datetime = sum(1 for v in datetime_vals if v is not None)
                if n_present > 0 and n_datetime / n_present > 0.95:
                    dates = [v for v in datetime_vals if v is not None]
                    mean, std = _robust_stats(dates)
                    self.column_stats[col] = ColumnStats(kind="datetime", mean=mean, std=std)
                    continue
                counts = Counter(str(v) for v in values if v not in (None, ""))
                cats = sorted(counts, key=lambda value: (-counts[value], value))[:_MAX_CATEGORIES]
                vocab = [_MISSING, _UNK] + cats
                self.column_stats[col] = ColumnStats(kind="categorical", vocab=vocab)

        target_vals = [r.get(target_column) for r in records]
        if task == "regression":
            nums = [v for v in (_try_float(v) for v in target_vals) if v is not None]
            self.target_mean, self.target_std = _robust_stats(nums) if nums else (0.0, 1.0)
        else:
            self.classes = sorted({str(v) for v in target_vals if v not in (None, "")})

        return self

    def transform(
        self, records: List[Dict[str, Any]], with_target: bool = True
    ) -> Dict[str, np.ndarray]:
        n = len(records)
        num_cols = self.numeric_columns
        cat_cols = self.categorical_columns

        numeric = np.zeros((n, max(len(num_cols), 1)), dtype=np.float32)
        categorical = np.zeros((n, max(len(cat_cols), 1)), dtype=np.int64)

        for i, r in enumerate(records):
            for j, col in enumerate(num_cols):
                stats = self.column_stats[col]
                v = _try_datetime(r.get(col)) if stats.kind == "datetime" else _try_float(r.get(col))
                if v is None:
                    v = stats.mean
                numeric[i, j] = (v - stats.mean) / stats.std
            for j, col in enumerate(cat_cols):
                stats = self.column_stats[col]
                raw = r.get(col)
                key = _MISSING if raw in (None, "") else str(raw)
                idx = stats.vocab.index(key) if key in stats.vocab else stats.vocab.index(_UNK)
                categorical[i, j] = idx

        out = {"numeric": numeric, "categorical": categorical}

        if with_target and self.target_column is not None:
            if self.task == "regression":
                target = np.zeros(n, dtype=np.float32)
                for i, r in enumerate(records):
                    v = _try_float(r.get(self.target_column))
                    v = self.target_mean if v is None else v
                    target[i] = (v - self.target_mean) / self.target_std
                out["target"] = target
            else:
                target = np.zeros(n, dtype=np.int64)
                for i, r in enumerate(records):
                    raw = str(r.get(self.target_column))
                    idx = self.classes.index(raw) if raw in self.classes else 0
                    target[i] = idx
                out["target"] = target

        return out

    def inverse_target(self, values: np.ndarray) -> List[Any]:
        if self.task == "regression":
            return [float(v * self.target_std + self.target_mean) for v in values]
        return [self.classes[int(v)] for v in values]

    def categorical_vocab_sizes(self) -> List[int]:
        return [len(self.column_stats[c].vocab) for c in self.categorical_columns]

    def state_dict(self) -> Dict:
        return {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "task": self.task,
            "column_stats": {
                c: {"kind": s.kind, "mean": s.mean, "std": s.std, "vocab": s.vocab}
                for c, s in self.column_stats.items()
            },
            "classes": self.classes,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "TabularPreprocessor":
        prep = cls(
            feature_columns=list(state["feature_columns"]),
            target_column=state["target_column"],
            task=state["task"],
            classes=list(state.get("classes", [])),
            target_mean=state.get("target_mean", 0.0),
            target_std=state.get("target_std", 1.0),
        )
        prep.column_stats = {
            c: ColumnStats(kind=v["kind"], mean=v.get("mean", 0.0), std=v.get("std", 1.0), vocab=list(v.get("vocab", [])))
            for c, v in state["column_stats"].items()
        }
        return prep
