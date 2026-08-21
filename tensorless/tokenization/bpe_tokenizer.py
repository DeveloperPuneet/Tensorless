"""A small, dependency-free byte-pair encoding tokenizer."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .char_tokenizer import BOS, EOS, PAD, SPECIAL_TOKENS, UNK


class BPETokenizer:
    """Character-initialized BPE tokenizer suitable for small local corpora.

    BPE operates on Unicode characters rather than bytes so the resulting
    vocabulary remains portable and decoding preserves the original text.
    """

    def __init__(self, vocab: List[str], merges: List[Tuple[str, str]]):
        self.vocab = list(vocab)
        self.merges = [tuple(pair) for pair in merges]
        self._stoi: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self._stoi[UNK]

    @property
    def bos_id(self) -> int:
        return self._stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self._stoi[EOS]

    @classmethod
    def build(
        cls, texts: Sequence[str], vocab_size: int = 1000, min_frequency: int = 2
    ) -> "BPETokenizer":
        if vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size must be at least {len(SPECIAL_TOKENS)}")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        symbols = set(char for text in texts for char in text)
        vocab = list(SPECIAL_TOKENS) + sorted(symbols - set(SPECIAL_TOKENS))
        sequences = [list(text) for text in texts]
        merges: List[Tuple[str, str]] = []

        while len(vocab) < vocab_size:
            counts = Counter(
                (sequence[index], sequence[index + 1])
                for sequence in sequences
                for index in range(len(sequence) - 1)
            )
            if not counts:
                break
            candidates = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            best_pair = next(
                (pair for pair, count in candidates
                 if count >= min_frequency and pair[0] + pair[1] not in vocab),
                None,
            )
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab.append(merged)
            sequences = [cls._apply_merge(sequence, best_pair, merged) for sequence in sequences]

        return cls(vocab, merges)

    @staticmethod
    def _apply_merge(sequence: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
        result: List[str] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def _tokenize(self, text: str) -> List[str]:
        tokens = list(text)
        for pair in self.merges:
            tokens = self._apply_merge(tokens, pair, pair[0] + pair[1])
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self._stoi.get(token, self.unk_id) for token in self._tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id} if skip_special_tokens else set()
        result = []
        for index in ids:
            if index in special_ids:
                continue
            if 0 <= index < len(self.vocab):
                token = self.vocab[index]
                if token == UNK:
                    result.append("\ufffd")
                elif token not in SPECIAL_TOKENS:
                    result.append(token)
        return "".join(result)

    def state_dict(self) -> Dict:
        return {
            "tokenizer_type": "bpe",
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "BPETokenizer":
        return cls(list(state["vocab"]), [tuple(pair) for pair in state.get("merges", [])])
