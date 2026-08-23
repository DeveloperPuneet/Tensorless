"""Early stopping.

Tracks a validation metric (lower is better, e.g. loss) and reports when
training should stop because it hasn't improved by at least `min_delta`
for `patience` consecutive checks.
"""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.best_state: dict = None
        self.num_bad_checks = 0
        self.should_stop = False

    def step(self, value: float, state: dict = None) -> bool:
        """Returns True if `value` is a new best."""
        if value < self.best - self.min_delta:
            self.best = value
            self.best_state = state
            self.num_bad_checks = 0
            return True
        else:
            self.num_bad_checks += 1
            if self.num_bad_checks >= self.patience:
                self.should_stop = True
            return False
