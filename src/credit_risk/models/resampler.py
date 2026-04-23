"""Resampling strategies for handling imbalanced datasets."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SMOTEResampler",
    "RandomOverSampler",
    "RandomUnderSampler",
    "create_resampler",
]


class Resampler(Protocol):
    """Protocol for resampling strategies."""

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Tuple of resampled (X, y)
        """
        ...


class SMOTEResampler:
    """SMOTE (Synthetic Minority Over-sampling Technique) resampler.

    Usage:
        resampler = SMOTEResampler(sampling_strategy="minority", k_neighbors=5)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    """

    def __init__(
        self,
        sampling_strategy: str = "minority",
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        """Initialize SMOTE resampler.

        Args:
            sampling_strategy: "minority", "majority", "all", or "not majority"
            k_neighbors: Number of nearest neighbors for synthetic generation
            random_state: Random seed
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the data using SMOTE.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of resampled (X, y)
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning("imbalanced-learn not installed, using random oversampling")
            fallback = RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
            return fallback.fit_resample(X, y)

        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(
            f"SMOTE resampled: {len(y)} -> {len(y_resampled)} "
            f"(class 0: {sum(y == 0)} -> {sum(y_resampled == 0)}, "
            f"class 1: {sum(y == 1)} -> {sum(y_resampled == 1)})"
        )
        return X_resampled, y_resampled


class RandomOverSampler:
    """Random oversampling of minority class.

    Usage:
        resampler = RandomOverSampler(sampling_strategy="minority")
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    """

    def __init__(
        self,
        sampling_strategy: str = "minority",
        random_state: int = 42,
    ):
        """Initialize random oversampler.

        Args:
            sampling_strategy: "minority", "majority", "all", or "not majority"
            random_state: Random seed
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample by randomly oversampling minority class.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of resampled (X, y)
        """
        from sklearn.utils import resample

        rng = np.random.default_rng(self.random_state)
        class_counts = np.bincount(y)
        n_minority = class_counts[0] if class_counts[0] < class_counts[1] else class_counts[1]
        n_majority = max(class_counts)

        if self.sampling_strategy == "minority":
            target_count = n_majority
        elif self.sampling_strategy == "majority":
            target_count = n_minority
        elif self.sampling_strategy == "all":
            target_count = n_majority
        else:
            target_count = n_majority

        X_resampled = [X]
        y_resampled = [y]

        for class_idx, count in enumerate(class_counts):
            if count < target_count:
                X_class = X[y == class_idx]
                n_to_add = target_count - count
                X_new = resample(
                    X_class,
                    replace=True,
                    n_samples=n_to_add,
                    random_state=rng.integers(2**31),
                )
                X_resampled.append(X_new)
                y_resampled.append(np.full(n_to_add, class_idx))

        X_out = np.vstack(X_resampled)
        y_out = np.concatenate(y_resampled)
        rng.shuffle(y_out)
        indices = rng.permutation(len(y_out))
        logger.info(
            f"Random oversampled: {len(y)} -> {len(y_out)} "
            f"(class 0: {sum(y == 0)} -> {sum(y_out == 0)}, "
            f"class 1: {sum(y == 1)} -> {sum(y_out == 1)})"
        )
        return X_out[indices], y_out[indices]


class RandomUnderSampler:
    """Random undersampling of majority class.

    Usage:
        resampler = RandomUnderSampler(sampling_strategy="majority")
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    """

    def __init__(
        self,
        sampling_strategy: str = "majority",
        random_state: int = 42,
    ):
        """Initialize random undersampler.

        Args:
            sampling_strategy: "majority", "minority", "all", or "not minority"
            random_state: Random seed
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample by randomly undersampling majority class.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of resampled (X, y)
        """
        from sklearn.utils import resample

        rng = np.random.default_rng(self.random_state)
        class_counts = np.bincount(y)
        n_minority = min(class_counts)
        n_majority = max(class_counts)

        if self.sampling_strategy == "majority":
            target_count = n_minority
        elif self.sampling_strategy == "minority":
            target_count = n_majority
        elif self.sampling_strategy == "all":
            target_count = n_minority
        else:
            target_count = n_minority

        X_resampled = []
        y_resampled = []

        for class_idx, count in enumerate(class_counts):
            X_class = X[y == class_idx]
            if count > target_count:
                X_new = resample(
                    X_class,
                    replace=False,
                    n_samples=target_count,
                    random_state=rng.integers(2**31),
                )
            else:
                X_new = X_class
            X_resampled.append(X_new)
            y_resampled.append(np.full(len(X_new), class_idx))

        X_out = np.vstack(X_resampled)
        y_out = np.concatenate(y_resampled)
        rng.shuffle(y_out)
        indices = rng.permutation(len(y_out))
        logger.info(
            f"Random undersampled: {len(y)} -> {len(y_out)} "
            f"(class 0: {sum(y == 0)} -> {sum(y_out == 0)}, "
            f"class 1: {sum(y == 1)} -> {sum(y_out == 1)})"
        )
        return X_out[indices], y_out[indices]


def create_resampler(
    method: str = "smote",
    **kwargs: Any,
) -> Resampler:
    """Create a resampler instance.

    Args:
        method: Resampling method ("smote", "over", "under", "none")
        **kwargs: Additional arguments for resampler

    Returns:
        Resampler instance
    """
    if method == "none" or method is None:
        return DummyResampler()

    if method == "smote":
        return SMOTEResampler(**kwargs)

    if method == "over":
        return RandomOverSampler(**kwargs)

    if method == "under":
        return RandomUnderSampler(**kwargs)

    raise ValueError(f"Unknown resampling method: {method}")


class DummyResampler:
    """Dummy resampler that returns data unchanged."""

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return data unchanged."""
        return X, y
