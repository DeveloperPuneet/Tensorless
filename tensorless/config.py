"""Configuration objects.

`TrainConfig` holds every knob a user can override in `tl.train(...)`.
Any field left as `None` means "let Tensorless decide automatically".
`ResolvedConfig` is what the auto-configuration system produces after
filling in every `None` with a concrete value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict


@dataclass
class TrainConfig:
    """User-facing training configuration. Every field is optional;
    unset fields are chosen automatically by the auto-config system.
    """

    # --- output / lifecycle ---
    out: Optional[str] = None                 # output .tl path, default "model.tl"
    force: bool = False                        # force retraining even if unchanged
    resume: Optional[bool] = None               # force/forbid resume (None = auto)
    ask_on_data_change: bool = False            # raise instead of auto-retrain on data change

    # --- task / architecture ---
    task: Optional[str] = None                  # "text-generation", "classification", "regression"
    model_type: Optional[str] = None            # "transformer", "mlp"
    d_model: Optional[int] = None
    layers: Optional[int] = None
    heads: Optional[int] = None
    ff_mult: Optional[int] = None
    dropout: Optional[float] = None
    gradient_checkpointing: Optional[bool] = None
    max_seq_len: Optional[int] = None
    tokenizer: Optional[str] = None           # "char" or "bpe"
    bpe_vocab_size: Optional[int] = None

    # --- optimization ---
    optimizer: Optional[str] = None             # "adamw", "adam", "sgd"
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    max_steps: Optional[int] = None
    grad_clip: Optional[float] = None
    warmup_steps: Optional[int] = None

    # --- validation / early stopping ---
    val_split: Optional[float] = None
    patience: Optional[int] = None
    min_delta: Optional[float] = None

    # --- hardware ---
    device: Optional[str] = None                # "cpu", "cuda", "tpu", or None = auto
    precision: Optional[str] = None              # "fp32", "fp16", "bf16"

    # --- checkpointing ---
    checkpoint_every: Optional[int] = None       # steps between checkpoints
    checkpoint_dir: Optional[str] = None
    checkpoint_shard_size_mb: Optional[int] = None  # 0/None keeps one checkpoint file

    # --- misc ---
    seed: int = 42
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def overrides(self) -> Dict[str, Any]:
        """Return only the fields the user explicitly set (non-None)."""
        d = asdict(self)
        d.pop("extra", None)
        return {k: v for k, v in d.items() if v is not None and v is not False}


@dataclass
class ResolvedConfig:
    """Fully resolved configuration -- every field has a concrete value.
    This is what actually gets used for training and what gets embedded
    in checkpoints and the final `.tl` file.
    """

    out: str
    force: bool
    resume: Optional[bool]
    ask_on_data_change: bool

    task: str
    model_type: str
    d_model: int
    layers: int
    heads: int
    ff_mult: int
    dropout: float
    gradient_checkpointing: bool
    max_seq_len: int
    tokenizer: str
    bpe_vocab_size: int

    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    max_steps: Optional[int]
    grad_clip: float
    warmup_steps: int

    val_split: float
    patience: int
    min_delta: float

    device: str
    precision: str

    checkpoint_every: int
    checkpoint_dir: str
    checkpoint_shard_size_mb: int

    seed: int
    verbose: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResolvedConfig":
        known = {f for f in cls.__dataclass_fields__.keys()}
        return cls(**{k: v for k, v in d.items() if k in known})
