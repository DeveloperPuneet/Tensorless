"""Character-level tokenizer.

Tensorless defaults to a character-level tokenizer because it:
  - requires no external vocabulary files or training corpus assumptions
  - works on any UTF-8 text out of the box (any language, code, etc.)
  - has a small, easily-portable vocabulary that fits directly inside a
    `.tl` file

This is deliberately simple rather than a full BPE tokenizer -- Tensorless
optimizes for "it just works with zero setup" over maximal efficiency.
Advanced users can plug in their own tokenizer via `model_type=` /
extension points documented in the developer docs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


class CharTokenizer:
    def __init__(self, vocab: List[str] = None):
        self.vocab: List[str] = vocab if vocab is not None else list(SPECIAL_TOKENS)
        self._stoi: Dict[str, int] = {c: i for i, c in enumerate(self.vocab)}

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
    def build(cls, texts: Sequence[str]) -> "CharTokenizer":
        chars = set()
        for t in texts:
            chars.update(t)
        vocab = list(SPECIAL_TOKENS) + sorted(chars)
        return cls(vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self._stoi.get(c, self.unk_id) for c in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id} if skip_special_tokens else set()
        for i in ids:
            if i in special_ids:
                continue
            if 0 <= i < len(self.vocab):
                tok = self.vocab[i]
                if tok == UNK:
                    chars.append("\ufffd")
                elif tok not in SPECIAL_TOKENS:
                    chars.append(tok)
        return "".join(chars)

    def state_dict(self) -> Dict:
        return {"tokenizer_type": "char", "vocab": self.vocab}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "CharTokenizer":
        return cls(vocab=list(state["vocab"]))
