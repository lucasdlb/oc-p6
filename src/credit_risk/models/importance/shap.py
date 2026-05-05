"""SHAP-based feature importance strategy."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np

from credit_risk.models.importance.base import BaseImportanceStrategy


class SHAPImportance(BaseImportanceStrategy):
    """Model-agnostic SHAP importance using KernelExplainer.

    Uses shap.Explainer with a callable to support any model type.
    """

    def __init__(
        self,
        n_samples: int = 500,
        random_state: int = 42,
    ):
        self._n_samples = n_samples
        self._random_state = random_state

    def _get_predict_fn(self, model: Any) -> Callable[[np.ndarray], np.ndarray]:
        """Get prediction function from model."""
        if hasattr(model, "predict_proba"):

            def predict_proba_positive(x: np.ndarray) -> np.ndarray:
                return model.predict_proba(x)[:, 1]

            return predict_proba_positive
        return model.predict

    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        """Compute SHAP importance on validation fold.

        Args:
            model: Trained model (used to get predict_proba)
            X: Validation features
            y: Validation targets (unused, kept for interface)
            model_params: Model hyperparameters (unused, kept for interface)

        Returns:
            SHAP-based importance scores
        """
        import shap

        if X.shape[0] > self._n_samples:
            rng = np.random.RandomState(self._random_state)
            idx = rng.choice(X.shape[0], self._n_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        X_sample = np.asarray(X_sample, dtype=np.float64)
        X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

        predict_fn = self._get_predict_fn(model)
        background = X_sample[: min(50, X_sample.shape[0])]
        explainer = shap.Explainer(predict_fn, background)
        result = explainer(X_sample)

        shap_values = result.values if hasattr(result, "values") else result

        return np.abs(cast(np.ndarray, shap_values)).mean(axis=0)
