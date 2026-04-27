"""Splitter protocol for cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Iterator, Protocol

import numpy as np

__all__ = ["Splitter", "TrainTestCVSplitter"]

from credit_risk.config import Config


class Splitter(Protocol):
    """Protocol for train/test splitting in cross-validation.

    Usage:
        class MySplitter:
            def split(self, X, y) -> Iterator[tuple[np.ndarray, np.ndarray]]:
                ...

        validator = CrossValidator(splitter=MySplitter(), ...)
    """

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) tuples.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Yields:
            Tuples of (train_indices, test_indices) for each fold.
        """
        ...


@dataclass
class TrainTestCVSplitter(Splitter):
    """Splitter that handles both train/test split AND CV folds on training set.

    This splitter:
    1. First splits data into train/test (holdout)
    2. Then creates CV folds ONLY on the training set

    Usage:
        splitter = TrainTestCVSplitter(
            test_size=0.2,
            n_splits=5,
            random_state=42,
            stratify=True
        )

        # Get train/test split
        X_train, X_test, y_train, y_test = splitter.split_train_test(X, y)

        # Get CV folds (on training set only)
        for train_idx, val_idx in splitter.split_cv(X_train, y_train):
            ...

    Attributes:
        test_size: Proportion of data for test set (0.0 to 1.0)
        n_splits: Number of folds for CV
        cv_random_state: Random seed
        stratify: Whether to stratify splits by target
    """

    test_size: float = 0.2
    n_splits: int = 5
    cv_random_state: int = 42
    test_random_state: int = 42
    stratify: bool = True
    shuffle: bool = True

    def __post_init__(self) -> None:
        from sklearn.model_selection import StratifiedKFold, train_test_split

        self._cv_splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.cv_random_state,
        )

        self._train_test_cv_splitter = partial(
            train_test_split,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.test_random_state,
        )

        self._resampler = None

    def set_resampler(self, resampler: Any) -> None:
        """Set a resampler to apply during resampling.

        Args:
            resampler: Resampler instance with fit_resample(X, y) method
        """
        self._resampler = resampler

    def split_train_test(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = self._train_test_cv_splitter(
            X,
            y,
            stratify=y if self.stratify else None,
        )

        if self._resampler is not None:
            X_train, y_train = self._resampler.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def split_cv(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Create CV folds on training data only.

        Args:
            X: Feature matrix (should be training data only)
            y: Target vector

        Yields:
            Tuples of (train_indices, val_indices) for each fold
        """
        return self._cv_splitter.split(X, y)

    @classmethod
    def from_config(cls, cfg: Config) -> TrainTestCVSplitter:
        return TrainTestCVSplitter(
            test_size=cfg.splitter.test_size,
            n_splits=cfg.splitter.n_splits,
            cv_random_state=cfg.splitter.cv_random_state,
            test_random_state=cfg.splitter.test_random_state,
            stratify=cfg.splitter.stratify,
            shuffle=cfg.splitter.shuffle,
        )
