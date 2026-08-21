from .bpe_tokenizer import BPETokenizer
from .char_tokenizer import CharTokenizer


def tokenizer_from_state_dict(state):
	if state.get("tokenizer_type", "char") == "bpe":
		return BPETokenizer.from_state_dict(state)
	return CharTokenizer.from_state_dict(state)


__all__ = ["CharTokenizer", "BPETokenizer", "tokenizer_from_state_dict"]
