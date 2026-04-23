"""Statistical feature importance strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

from credit_risk.models.importance.base import BaseImportanceStrategy


class StatisticalImportance(BaseImportanceStrategy):
    """Feature importance using statistical tests.

    This is a univariate method - it doesn't need a trained model.
    It computes feature importance directly from X and y.

    Combines multiple univariate tests:
    - Chi-squared (for categorical/binary)
    - ANOVA F-test
    - Pearson correlation

    Normalizes each score to [0, 1] and returns mean across methods.
    Features passing threshold in at least min_tests methods are selected.
    """

    def __init__(
        self,
        min_tests: int = 1,
        threshold: float = 0.1,
        random_state: int = 42,
    ):
        self._min_tests = min_tests
        self._threshold = threshold
        self._random_state = random_state

    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        """Compute statistical importance (univariate - no model needed).

        This method is called per fold but computes same scores since
        it's a univariate method independent of the model.

        Args:
            model: Ignored (kept for interface compatibility)
            X: Feature matrix for current fold
            y: Target vector for current fold
            model_params: Ignored (kept for interface)

        Returns:
            Statistical importance scores
        """
        return self._compute_scores(X, y)

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Internal method to compute statistical scores."""
        scores_per_method: list[np.ndarray] = []

        X_positive = self._make_non_negative(X)
        if X_positive is not None:
            chi2_scores, _ = chi2(X_positive, y)
            chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
            chi2_scores = self._normalize_scores(chi2_scores)
            scores_per_method.append(chi2_scores)

        f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        f_scores = self._normalize_scores(f_scores)
        scores_per_method.append(f_scores)

        corr_scores = self._correlation_scores(X, y)
        corr_scores = self._normalize_scores(corr_scores)
        scores_per_method.append(corr_scores)

        stacked = np.vstack(scores_per_method)

        passed_tests = np.sum(stacked >= self._threshold, axis=0)
        combined_scores = np.mean(stacked, axis=0)

        final_scores = np.where(passed_tests >= self._min_tests, combined_scores, 0.0)

        return final_scores

    def _make_non_negative(self, X: np.ndarray) -> np.ndarray | None:
        X_min = X.min()
        if X_min >= 0:
            return X
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        min_val, max_val = scores.min(), scores.max()
        if max_val - min_val > 0:
            return (scores - min_val) / (max_val - min_val)
        return np.zeros_like(scores)

    def _correlation_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_float = X.astype(np.float64)
        y_float = y.astype(np.float64)

        correlations = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            feature_col = X_float[:, i]
            valid_mask = np.isfinite(feature_col) & np.isfinite(y_float)
            if valid_mask.sum() > 1 and np.std(feature_col[valid_mask]) > 0:
                corr = np.corrcoef(feature_col[valid_mask], y_float[valid_mask])
                if corr.shape == (2, 2):
                    correlations[i] = abs(corr[0, 1])
        return np.nan_to_num(correlations)
