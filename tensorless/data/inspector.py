"""Dataset inspection: `tl.inspect("./data")`.

Loads the dataset, detects its task type, and reports size, samples,
detected problems, and recommendations -- without training anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .loader import load_dataset, Dataset
from .fingerprint import fingerprint_path


@dataclass
class InspectionReport:
    path: str
    fingerprint: str
    kind: str
    task: str
    n_examples: int
    n_files: int
    columns: List[str] = field(default_factory=list)
    sample: Any = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Dataset: {self.path}",
            f"  fingerprint : {self.fingerprint[:16]}...",
            f"  detected kind : {self.kind}",
            f"  detected task : {self.task}",
            f"  examples      : {self.n_examples}",
            f"  files         : {self.n_files}",
        ]
        if self.columns:
            lines.append(f"  columns       : {', '.join(self.columns)}")
        for k, v in self.stats.items():
            lines.append(f"  {k:14s}: {v}")
        if self.warnings:
            lines.append("  Warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        if self.recommendations:
            lines.append("  Recommendations:")
            lines += [f"    - {r}" for r in self.recommendations]
        return "\n".join(lines)


def _compute_stats(ds: Dataset) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if ds.kind in ("text", "text_labeled"):
        lengths = [len(t) for t in ds.texts]
        if lengths:
            stats["avg_chars"] = round(sum(lengths) / len(lengths), 1)
            stats["min_chars"] = min(lengths)
            stats["max_chars"] = max(lengths)
        if ds.kind == "text_labeled":
            classes = sorted(set(ds.labels))
            stats["n_classes"] = len(classes)
            stats["classes"] = classes[:20]
    else:
        stats["n_rows"] = len(ds.records)
        stats["n_columns"] = len(ds.columns)
    return stats


def _warnings_and_recommendations(ds: Dataset, task: str) -> (List[str], List[str]):
    warnings: List[str] = []
    recs: List[str] = []
    n = len(ds)

    if n == 0:
        warnings.append("Dataset is empty.")
        return warnings, recs

    if n < 20:
        warnings.append(
            f"Only {n} example(s) found. This is very small for training a "
            f"useful model; results may be poor / mostly memorization."
        )
        recs.append("Collect more data if possible (aim for hundreds+ examples).")

    if ds.kind == "text_labeled":
        from collections import Counter

        counts = Counter(ds.labels)
        if len(counts) < 2:
            warnings.append("Only one class detected; classification needs 2+ classes.")
        else:
            majority = max(counts.values())
            minority = min(counts.values())
            if majority > 3 * max(minority, 1):
                warnings.append(
                    f"Class imbalance detected (largest class {majority} vs "
                    f"smallest {minority}). Consider balancing or using "
                    f"class weights."
                )
                recs.append(
                    "Tensorless will still train, but consider collecting more "
                    "examples for underrepresented classes."
                )

    if ds.kind == "tabular":
        missing_cols = set()
        for r in ds.records[:200]:
            for c in ds.columns:
                v = r.get(c, "")
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    missing_cols.add(c)
        if missing_cols:
            warnings.append(
                f"Missing values detected in column(s): {', '.join(sorted(missing_cols))}."
            )
            recs.append(
                "Tensorless will impute missing numeric values with the column "
                "mean and missing categorical values with a placeholder token."
            )

    if ds.kind in ("text", "text_labeled"):
        avg_len = sum(len(t) for t in ds.texts) / max(1, len(ds.texts))
        if avg_len > 20000:
            recs.append(
                "Texts are long; Tensorless will truncate to the configured "
                "max_seq_len. Pass max_seq_len=... to change this."
            )

    return warnings, recs


def inspect_path(path: str) -> InspectionReport:
    # Local import to avoid a circular import between `data` and `auto`.
    from ..auto.detector import detect_task

    ds = load_dataset(path)
    task = detect_task(ds)
    fp = fingerprint_path(path)
    stats = _compute_stats(ds)
    warnings, recs = _warnings_and_recommendations(ds, task)

    sample: Any = None
    if ds.kind in ("text", "text_labeled") and ds.texts:
        sample = ds.texts[0][:300]
    elif ds.records:
        sample = ds.records[0]

    return InspectionReport(
        path=path,
        fingerprint=fp,
        kind=ds.kind,
        task=task,
        n_examples=len(ds),
        n_files=ds.n_files,
        columns=list(ds.columns),
        sample=sample,
        warnings=warnings,
        recommendations=recs,
        stats=stats,
    )
