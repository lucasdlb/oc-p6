"""Splitter protocol for cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Any

import numpy as np

__all__ = ["Splitter", "TrainTestCVSplitter"]


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
        random_state: Random seed
        stratify: Whether to stratify splits by target
    """

    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    stratify: bool = True

    def __post_init__(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        self._cv_splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
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
        rng = np.random.default_rng(self.random_state)
        n_samples = len(y)

        if self.stratify:
            # Stratified split
            unique_classes, class_indices = np.unique(y, return_inverse=True)
            train_indices = []
            test_indices = []

            for c in unique_classes:
                class_mask = class_indices == c
                class_indices_orig = np.where(class_mask)[0]
                rng.shuffle(class_indices_orig)

                n_test = int(len(class_indices_orig) * self.test_size)
                test_indices.extend(class_indices_orig[:n_test])
                train_indices.extend(class_indices_orig[n_test:])

            train_idx = np.array(train_indices)
            test_idx = np.array(test_indices)
        else:
            # Simple random split
            indices = rng.permutation(n_samples)
            n_test = int(n_samples * self.test_size)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        X_train = X[train_idx]
        y_train = y[train_idx]

        if self._resampler is not None:
            X_train, y_train = self._resampler.fit_resample(X_train, y_train)

        return (
            X_train,
            X[test_idx],
            y_train,
            y[test_idx],
        )

    def split_cv(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Create CV folds on training data only.

        Args:
            X: Feature matrix (should be training data only)
            y: Target vector

        Yields:
            Tuples of (train_indices, val_indices) for each fold
        """
        return self._cv_splitter.split(X, y)
