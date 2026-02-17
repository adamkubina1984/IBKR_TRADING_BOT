# model/tscv.py
from collections.abc import Iterator

import numpy as np


class PurgedWalkForwardSplit:
    def __init__(self, n_splits=5, embargo=10):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = fold*(k+1)
            test_start = train_end + self.embargo
            test_end = min(test_start + fold, n)
            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(test_start, test_end)
            yield train_idx, test_idx
