"""Model predictor for inference."""

import numpy as np
from sklearn.base import BaseEstimator


class ModelPredictor:
    def __init__(self, model: BaseEstimator | None = None):
        self.model = model

    def set_model(self, model: BaseEstimator) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> np.ndarray | None:
        if self.model is None:
            raise ValueError("No model set. Call set_model() first.")
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
