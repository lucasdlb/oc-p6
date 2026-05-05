"""Base class for importance strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseImportanceStrategy(ABC):
    """Base class for importance strategies.

    Subclasses must implement:
    - compute_fold(): Compute importance for a single fold
    - aggregate(): Aggregate fold importances (default: mean)

    Example:
        class MyImportance(BaseImportanceStrategy):
            def compute_fold(self, model, X, y, model_params):
                return model.feature_importances_

            def aggregate(self, fold_importances):
                return np.mean(fold_importances, axis=0)
    """

    @abstractmethod
    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        """Compute importance scores for a single fold.

        Called by CrossValidator on each validation fold after model is trained.

        Args:
            model: Trained model from current fold
            X: Validation feature matrix (n_samples, n_features)
            y: Validation target vector
            model_params: Model hyperparameters

        Returns:
            Importance scores (n_features,)
        """
        ...

    def aggregate(self, fold_importances: list[np.ndarray]) -> np.ndarray:
        """Aggregate importance scores from all folds.

        Default: mean across folds.

        Args:
            fold_importances: List of importance arrays, one per fold

        Returns:
            Aggregated importance scores (n_features,)
        """
        return np.mean(fold_importances, axis=0)
