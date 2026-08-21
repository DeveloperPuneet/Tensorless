"""Turns a loaded `Dataset` + resolved config into PyTorch-ready tensors,
train/val splits, and the tokenizer/preprocessor that produced them.

Kept separate from `trainer.py` so the "how do I turn data into tensors"
logic can be tested and extended independently of the training loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from ..data.loader import Dataset
from ..data.tabular import TabularPreprocessor
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..tokenization.char_tokenizer import CharTokenizer
from ..auto.detector import target_column
from ..errors import DataError


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    meta: Dict[str, Any]
    tokenizer: Optional[CharTokenizer] = None
    preprocessor: Optional[TabularPreprocessor] = None


class _LMChunkDataset(TorchDataset):
    def __init__(self, token_ids: List[int], block_size: int):
        self.ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class _ClsTextDataset(TorchDataset):
    def __init__(self, input_ids: List[List[int]], attn_masks: List[List[int]], labels: List[int]):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.attn_masks[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class _TabularDataset(TorchDataset):
    def __init__(self, numeric: torch.Tensor, categorical: torch.Tensor, target: torch.Tensor):
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
    all_ids: List[int] = []
    for t in ds.texts:
        all_ids.extend(tokenizer.encode(t, add_special_tokens=True))

    block_size = cfg["max_seq_len"]
    if len(all_ids) <= block_size:
        # Pad tiny corpora so we have at least one training example.
        all_ids = all_ids + [tokenizer.pad_id] * (block_size + 1 - len(all_ids))

    n_val_tokens = int(len(all_ids) * cfg["val_split"])
    if n_val_tokens > block_size + 1:
        split_point = len(all_ids) - n_val_tokens
        train_ids, val_ids = all_ids[:split_point], all_ids[split_point:]
    else:
        train_ids, val_ids = all_ids, None

    train_ds = _LMChunkDataset(train_ids, block_size)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False
    )

    val_loader = None
    if val_ids is not None:
        val_ds = _LMChunkDataset(val_ids, block_size)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

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

    train_loader = DataLoader(subset(train_idx), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = (
        DataLoader(subset(val_idx), batch_size=cfg["batch_size"], shuffle=False) if val_idx else None
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
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)

    train_ds = _TabularDataset(
        transformed["numeric"][train_idx_t],
        transformed["categorical"][train_idx_t],
        transformed["target"][train_idx_t],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    val_loader = None
    if val_idx:
        val_idx_t = torch.tensor(val_idx, dtype=torch.long)
        val_ds = _TabularDataset(
            transformed["numeric"][val_idx_t],
            transformed["categorical"][val_idx_t],
            transformed["target"][val_idx_t],
        )
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    meta = {
        "n_numeric": len(prep.numeric_columns),
        "categorical_vocab_sizes": prep.categorical_vocab_sizes(),
        "n_classes": len(prep.classes) if task == "classification" else 0,
    }
    return PreparedData(train_loader=train_loader, val_loader=val_loader, meta=meta, preprocessor=prep)
