"""Turns a loaded `Dataset` + resolved config into PyTorch-ready tensors,
train/val splits, and the tokenizer/preprocessor that produced them.

Kept separate from `trainer.py` so the "how do I turn data into tensors"
logic can be tested and extended independently of the training loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.loader import Dataset
from ..data.tabular import TabularPreprocessor
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..tokenization.char_tokenizer import CharTokenizer
from ..auto.detector import target_column
from ..errors import DataError


@dataclass
class PreparedData:
    train_loader: "BatchLoader"
    val_loader: Optional["BatchLoader"]
    meta: Dict[str, Any]
    tokenizer: Optional[CharTokenizer] = None
    preprocessor: Optional[TabularPreprocessor] = None


class BatchLoader:
    def __init__(self, dataset, batch_size, shuffle=False, seed=0):
        self.dataset, self.batch_size, self.shuffle, self.seed = dataset, batch_size, shuffle, seed
        self.epoch = 0
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        if not hasattr(self.dataset, "__getitem__"):
            batch = []
            for value in self.dataset:
                batch.append(value)
                if len(batch) == self.batch_size:
                    yield tuple(np.stack(items) for items in zip(*batch))
                    batch = []
            if batch:
                yield tuple(np.stack(items) for items in zip(*batch))
            return
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
            self.epoch += 1
        for start in range(0, len(indices), self.batch_size):
            values = [self.dataset[i] for i in indices[start:start + self.batch_size]]
            yield tuple(np.stack(items) for items in zip(*values))


class _StreamingLMChunkDataset:
    """Tokenize text lazily and yield fixed-size language-model batches."""

    def __init__(self, texts: List[str], tokenizer, block_size: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self._length = sum(self._count_chunks(text) for text in texts)

    def _count_chunks(self, text: str) -> int:
        token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
        return max(1, token_count - self.block_size)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for text in self.texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.block_size:
                ids = ids + [self.tokenizer.pad_id] * (self.block_size + 1 - len(ids))
            for start in range(0, len(ids) - self.block_size):
                chunk = ids[start: start + self.block_size + 1]
                if len(chunk) != self.block_size + 1:
                    continue
                yield (
                    np.asarray(chunk[:-1], dtype=np.int64),
                    np.asarray(chunk[1:], dtype=np.int64),
                )


class _ClsTextDataset:
    def __init__(self, input_ids: List[List[int]], attn_masks: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            np.asarray(self.input_ids[idx], dtype=np.int64),
            np.asarray(self.attn_masks[idx], dtype=np.float32),
            np.asarray(self.labels[idx], dtype=np.int64),
        )


class _TabularDataset:
    def __init__(self, numeric, categorical, target):
        self.numeric = numeric
        self.categorical = categorical
        self.target = target

    def __len__(self) -> int:
        return self.numeric.shape[0]

    def __getitem__(self, idx: int):
        return self.numeric[idx], self.categorical[idx], self.target[idx]


def _split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if not train_idx:
        train_idx = idx
        val_idx = []
    return train_idx, val_idx


def _build_tokenizer(ds: Dataset, cfg: Dict[str, Any]):
    if cfg.get("tokenizer", "bpe") == "bpe":
        return BPETokenizer.build(ds.texts, vocab_size=cfg.get("bpe_vocab_size", 1000))
    return CharTokenizer.build(ds.texts)


def prepare_text_generation(
    ds: Dataset, cfg: Dict[str, Any], tokenizer=None
) -> PreparedData:
    tokenizer = tokenizer or _build_tokenizer(ds, cfg)
    block_size = cfg["max_seq_len"]
    n_val_texts = int(len(ds.texts) * cfg["val_split"])
    if n_val_texts and n_val_texts < len(ds.texts):
        train_texts = ds.texts[:-n_val_texts]
        val_texts = ds.texts[-n_val_texts:]
    elif cfg["val_split"] > 0 and len(ds.texts) == 1:
        text = ds.texts[0]
        split_point = int(len(text) * (1.0 - cfg["val_split"]))
        if split_point >= block_size:
            train_texts = [text[:split_point]]
            val_texts = [text[split_point:]]
        else:
            train_texts, val_texts = ds.texts, None
    else:
        train_texts, val_texts = ds.texts, None

    train_ds = _StreamingLMChunkDataset(train_texts, tokenizer, block_size)
    train_loader = BatchLoader(train_ds, cfg["batch_size"])

    val_loader = None
    if val_texts is not None:
        val_ds = _StreamingLMChunkDataset(val_texts, tokenizer, block_size)
        if len(val_ds) > 0:
            val_loader = BatchLoader(val_ds, cfg["batch_size"])

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "n_classes": 0,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, tokenizer=tokenizer)


def prepare_text_classification(
    ds: Dataset,
    cfg: Dict[str, Any],
    tokenizer=None,
    classes: Optional[List[str]] = None,
) -> PreparedData:
    tokenizer = tokenizer or _build_tokenizer(ds, cfg)
    classes = classes or sorted(set(ds.labels))
    label2id = {c: i for i, c in enumerate(classes)}

    block_size = cfg["max_seq_len"]
    input_ids, attn_masks, labels = [], [], []
    for text, label in zip(ds.texts, ds.labels):
        ids = tokenizer.encode(text, add_special_tokens=True)[:block_size]
        mask = [1] * len(ids)
        if len(ids) < block_size:
            pad_len = block_size - len(ids)
            ids = ids + [tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len
        input_ids.append(ids)
        attn_masks.append(mask)
        labels.append(label2id.get(label, 0))

    n = len(labels)
    train_idx, val_idx = _split_indices(n, cfg["val_split"], cfg["seed"])

    def subset(indices):
        return _ClsTextDataset(
            [input_ids[i] for i in indices],
            [attn_masks[i] for i in indices],
            [labels[i] for i in indices],
        )

    train_loader = BatchLoader(subset(train_idx), cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    val_loader = (
        BatchLoader(subset(val_idx), cfg["batch_size"]) if val_idx else None
    )

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "n_classes": len(classes),
        "classes": classes,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, tokenizer=tokenizer)


def prepare_tabular(
    ds: Dataset,
    cfg: Dict[str, Any],
    task: str,
    preprocessor: Optional[TabularPreprocessor] = None,
) -> PreparedData:
    target_col = target_column(ds)
    if target_col is None:
        raise DataError("Could not determine a target column for tabular training.")

    prep = preprocessor or TabularPreprocessor().fit(ds.records, ds.columns, target_col, task)
    transformed = prep.transform(ds.records, with_target=True)

    n = transformed["numeric"].shape[0]
    train_idx, val_idx = _split_indices(n, cfg["val_split"], cfg["seed"])
    train_ds = _TabularDataset(
        transformed["numeric"][train_idx],
        transformed["categorical"][train_idx],
        transformed["target"][train_idx],
    )
    train_loader = BatchLoader(train_ds, cfg["batch_size"], shuffle=True, seed=cfg["seed"])

    val_loader = None
    if val_idx:
        val_ds = _TabularDataset(
            transformed["numeric"][val_idx],
            transformed["categorical"][val_idx],
            transformed["target"][val_idx],
        )
        val_loader = BatchLoader(val_ds, cfg["batch_size"])

    meta = {
        "n_numeric": len(prep.numeric_columns),
        "categorical_vocab_sizes": prep.categorical_vocab_sizes(),
        "n_classes": len(prep.classes) if task == "classification" else 0,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, preprocessor=prep)
