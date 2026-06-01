"""SHAP-based feature importance strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from credit_risk.models.importance.base import BaseImportanceStrategy


class SHAPImportance(BaseImportanceStrategy):
    """Feature importance via SHAP values.

    Strategy selection:
    - Tree-based models (LightGBM, XGBoost, RandomForest, etc.) use
      ``shap.TreeExplainer`` — exact, fast, no sample-size constraints.
    - All other models use ``shap.PermutationExplainer`` with
      ``max_evals = 2 * n_features + 1`` (the minimum required by SHAP).

    Args:
        n_background: Number of background samples passed to PermutationExplainer
            as the masker.  Ignored for TreeExplainer.
        n_samples: Maximum number of validation rows to explain.  Larger values
            give more stable estimates at higher cost.
        random_state: Seed for reproducible subsetting.
    """

    def __init__(
        self,
        n_background: int = 50,
        n_samples: int = 200,
        random_state: int = 42,
    ):
        self._n_background = n_background
        self._n_samples = n_samples
        self._random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        """Compute mean |SHAP| importance on a validation fold.

        Args:
            model: Trained sklearn-compatible model or pipeline.
            X: Validation feature matrix ``(n_samples, n_features)``.
            y: Validation targets — unused, kept for interface compatibility.
            model_params: Model hyperparameters — unused.

        Returns:
            Importance scores ``(n_features,)`` — mean absolute SHAP value
            per feature.
        """
        X_clean = self._prepare(X)
        X_sample = self._subsample(X_clean)

        estimator = self._unwrap(model)

        if self._is_tree_model(estimator):
            shap_values = self._tree_shap(estimator, X_sample)
        else:
            shap_values = self._permutation_shap(model, X_sample)

        return np.abs(shap_values).mean(axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(X: np.ndarray) -> np.ndarray:
        """Cast to float64 and replace non-finite values."""
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        """Randomly subsample rows if X is larger than n_samples."""
        if X.shape[0] <= self._n_samples:
            return X
        rng = np.random.RandomState(self._random_state)
        idx = rng.choice(X.shape[0], self._n_samples, replace=False)
        return X[idx]

    @staticmethod
    def _unwrap(model: Any) -> Any:
        """Unwrap sklearn Pipeline to get the final estimator."""
        if hasattr(model, "steps"):
            return model.steps[-1][1]
        return model

    @staticmethod
    def _is_tree_model(estimator: Any) -> bool:
        """Return True if the estimator is a tree-based model supported by TreeExplainer."""
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            from xgboost import XGBClassifier, XGBRegressor

            tree_types = (LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor)
        except ImportError:
            tree_types = ()

        try:
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
                RandomForestClassifier,
            )

            tree_types = tree_types + (  # type: ignore[assignment]
                RandomForestClassifier,
                ExtraTreesClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
            )
        except ImportError:
            pass

        return isinstance(estimator, tree_types)

    def _tree_shap(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values via TreeExplainer (fast, exact)."""
        import shap

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X, check_additivity=False)

        # LightGBM binary classification returns a single array; multiclass
        # returns a list — take the positive class or average.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)

        return np.asarray(shap_values, dtype=np.float64)

    def _permutation_shap(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values via PermutationExplainer (model-agnostic)."""
        import shap

        n_features = X.shape[1]
        # PermutationExplainer requires max_evals >= 2 * n_features + 1
        max_evals = 2 * n_features + 1

        def predict_fn(x: np.ndarray) -> np.ndarray:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(x)[:, 1]
            return model.predict(x)

        background = X[: min(self._n_background, X.shape[0])]
        explainer = shap.PermutationExplainer(predict_fn, background)
        result = explainer(X, max_evals=max_evals)

        shap_values = result.values if hasattr(result, "values") else result
        return np.asarray(shap_values, dtype=np.float64)
