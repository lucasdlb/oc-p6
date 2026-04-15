"""SHAP explainer for model interpretability."""

from __future__ import annotations

import numpy as np
import shap
from sklearn.base import BaseEstimator

from credit_risk.config import Config


class ShapExplainer:
    def __init__(self, config: Config | None = None):
        from credit_risk.config import cfg

        self.config = config or cfg
        self.explainer: shap.Explainer | None = None
        self.expected_value: np.ndarray | None = None

    def fit(self, model: BaseEstimator, X_background: np.ndarray) -> "ShapExplainer":
        background = shap.sample(
            X_background, self.config.interpret.shap_background_samples, random_state=42
        )
        self.explainer = shap.TreeExplainer(model, data=background)
        return self

    def global_importance(
        self, X: np.ndarray, n_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        n = n_samples or self.config.interpret.shap_n_samples
        X_sample = shap.sample(X, min(n, X.shape[0]), random_state=42)
        shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return shap_values, mean_abs

    def local_importance(
        self, X_instance: np.ndarray, n_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        n = n_samples or self.config.interpret.shap_n_samples
        X_sample = shap.sample(X_instance, min(n, X_instance.shape[0]), random_state=42)
        shap_values = self.explainer.shap_values(X_sample)
        base_value = self.explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = np.array(base_value)
        else:
            base_value = np.array([base_value])
        return shap_values, base_value

    def feature_contributions(
        self, X_instance: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        X_single = X_instance[:1]
        shap_values = self.explainer.shap_values(X_single)
        contributions = shap_values[0]
        ranked = sorted(enumerate(contributions), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:top_k]
