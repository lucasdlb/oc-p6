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
        """Compute permutation importance on validation fold.

        Uses the already-fitted model from the current CV fold.

        Args:
            model: Trained sklearn-compatible model or pipeline.
            X: Validation feature matrix ``(n_samples, n_features)``.
            y: Validation targets.
            model_params: Model hyperparameters — unused.

        Returns:
            Importance scores ``(n_features,)`` — mean decrease in ROC AUC.
        """
        # Replace non-finite values — ratio features can produce inf/nan
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Our EstimatorPipeline.predict_proba returns 1D (positive class only).
        # sklearn's built-in "roc_auc" scorer expects 2D for binary classification.
        # Build a custom scorer using a callable that calls our pipeline directly.
        from sklearn.metrics import roc_auc_score

        def _scorer(estimator, X_, y_):
            y_prob = estimator.predict_proba(X_)
            return roc_auc_score(y_, y_prob)

        scores = permutation_importance(
            model,
            X,
            y,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            scoring=_scorer,
        )
        return scores.importances_mean
