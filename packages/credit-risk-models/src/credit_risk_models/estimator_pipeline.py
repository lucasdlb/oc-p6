"""EstimatorPipeline — sklearn Pipeline subclass with predict_proba for classification."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

ESTIMATOR_STEP = "estimator"


class EstimatorPipeline(Pipeline):
    """Pipeline with nan_replace, x_transform, and estimator steps.

    Provides utility methods for accessing the underlying estimator,
    prediction methods, and feature importances.
    """

    def get_final_estimator(self) -> BaseEstimator:
        """Return the actual estimator (unwrapped from any wrapper)."""
        return self.named_steps[ESTIMATOR_STEP]

    def predict_proba(self, X, **params):
        """Predict class probabilities, returning positive-class scores (1D).

        Handles LightGBM's cross_entropy_lambda objective which returns raw
        logit scores instead of probabilities — applies sigmoid when output
        exceeds the [0, 1] range.

        Falls back to decision_function for estimators without predict_proba
        (e.g. RidgeClassifier, SVC with probability=False).
        """
        try:
            result = super().predict_proba(X, **params)
            if result.max() > 1.0 or result.min() < 0.0:
                from scipy.special import expit

                result = expit(result)
            return result[:, 1]
        except AttributeError:
            from scipy.special import expit

            scores = super().decision_function(X, **params)
            return expit(scores)

    def get_feature_importances(self):
        """Get feature importances from the estimator.

        Handles both tree-based models (feature_importances_) and
        linear models (abs(coef_) averaged across classes).

        Returns:
            Feature importances as numpy array, or None if not available.
        """
        estimator = self.get_final_estimator()
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        return None
