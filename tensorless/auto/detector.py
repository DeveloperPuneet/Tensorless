"""Automatic task-type detection.

Given a loaded `Dataset`, decide what kind of ML task it represents:

  - "text-generation"  : raw text corpus -> train a language model
  - "text-classification": (text, label) pairs -> classify text
  - "classification"    : tabular data with a categorical target
  - "regression"        : tabular data with a numeric target

The detector is intentionally simple and explainable: every decision can
be described in one sentence, which is important because `tl.inspect()`
reports *why* a task was chosen.
"""

from __future__ import annotations

from typing import Optional

from ..data.loader import Dataset
from ..errors import DataError

_TARGET_CANDIDATES = ("label", "target", "class", "category", "y", "output")


def _looks_numeric(values) -> bool:
    n_ok = 0
    n_total = 0
    for v in values:
        if v is None or v == "":
            continue
        n_total += 1
        try:
            float(v)
            n_ok += 1
        except (TypeError, ValueError):
            pass
    if n_total == 0:
        return False
    return (n_ok / n_total) > 0.95


def _find_target_column(ds: Dataset) -> Optional[str]:
    for cand in _TARGET_CANDIDATES:
        for col in ds.columns:
            if col.lower() == cand:
                return col
    # Fall back to the last column, a common convention in tabular datasets.
    return ds.columns[-1] if ds.columns else None


def detect_task(ds: Dataset) -> str:
    """Return one of "text-generation", "text-classification",
    "classification", "regression".
    """
    if ds.kind == "text":
        return "text-generation"

    if ds.kind == "text_labeled":
        return "text-classification"

    if ds.kind == "tabular":
        if not ds.records:
            raise DataError("Tabular dataset has no rows.")
        target = _find_target_column(ds)
        if target is None:
            raise DataError(
                "Could not find a target/label column in the tabular data. "
                "Tensorless looks for a column named one of: "
                f"{', '.join(_TARGET_CANDIDATES)} (or uses the last column)."
            )
        values = [r.get(target) for r in ds.records]
        if _looks_numeric(values):
            unique_vals = set(values)
            # Small number of distinct numeric values that are all
            # integer-like -> more likely classification (e.g. 0/1 labels)
            # than regression.
            if len(unique_vals) <= min(10, max(2, len(values) // 20)):
                try:
                    all_int_like = all(float(v) == int(float(v)) for v in unique_vals if v not in (None, ""))
                except (TypeError, ValueError):
                    all_int_like = False
                if all_int_like:
                    return "classification"
            return "regression"
        return "classification"

    raise DataError(f"Unknown dataset kind: {ds.kind}")


def target_column(ds: Dataset) -> Optional[str]:
    """Public helper mirroring `_find_target_column`, used by the tabular
    model pipeline so detection and training agree on which column is
    the target.
    """
    return _find_target_column(ds)
