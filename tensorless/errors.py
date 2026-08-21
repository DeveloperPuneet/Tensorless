"""Tensorless error hierarchy.

All errors raised by Tensorless inherit from :class:`TensorlessError`, so
callers can do::

    try:
        tl.train("./data")
    except tl.TensorlessError as e:
        ...

instead of catching bare Exception.
"""


class TensorlessError(Exception):
    """Base class for all Tensorless errors."""


class DataError(TensorlessError):
    """Raised when a dataset cannot be read, is empty, or is malformed."""


class ConfigError(TensorlessError):
    """Raised when user-supplied configuration is invalid or contradictory."""


class ModelError(TensorlessError):
    """Raised for unsupported model/task combinations or model build failures."""


class CheckpointError(TensorlessError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


class SerializationError(TensorlessError):
    """Raised when a `.tl` file cannot be written or read."""
