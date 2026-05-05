"""Permutation-based feature importance strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from credit_risk.models.importance.base import BaseImportanceStrategy


class PermutationImportance(BaseImportanceStrategy):
    """Feature importance using sklearn permutation importance.

    Measures decrease in model performance when feature values are shuffled.
    """

    def __init__(
        self,
        n_repeats: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self._n_repeats = n_repeats
        self._random_state = random_state
        self._n_jobs = n_jobs

    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        scores = permutation_importance(
            model,
            X,
            y,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            scoring="roc_auc",
        )
        return scores.importances_mean
        """Compute permutation importance on validation fold.

        Uses the already-fitted model from the current CV fold.
        """
        scores = permutation_importance(
            model,
            X,
            y,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            scoring="roc_auc",
        )

        return scores.importances_mean
