"""
Tensorless
==========

ML with maximum automation and minimum setup.

    import tensorless as tl

    tl.train("./data")
    tl.run("model.tl")

See https://github.com/tensorless/tensorless for full documentation.
"""

from .api import train, run, load, inspect
from .config import TrainConfig
from .errors import (
    TensorlessError,
    DataError,
    ConfigError,
    ModelError,
    CheckpointError,
    SerializationError,
)
from ._version import __version__

__all__ = [
    "train",
    "run",
    "load",
    "inspect",
    "TrainConfig",
    "TensorlessError",
    "DataError",
    "ConfigError",
    "ModelError",
    "CheckpointError",
    "SerializationError",
    "__version__",
]
