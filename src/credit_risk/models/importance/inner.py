"""Inner model-based feature importance strategy.

Uses model's internal feature_importances_ (e.g., LightGBM, RandomForest, XGBoost).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from credit_risk.models.importance.base import BaseImportanceStrategy


class InnerImportance(BaseImportanceStrategy):
    """Feature importance using model internal feature_importances_.

    Works with any model that has feature_importances_ attribute:
    - LightGBM (LGBMClassifier)
    - RandomForest (RandomForestClassifier)
    - XGBoost (XGBClassifier)
    - CatBoost (CatBoostClassifier)
    """

    def compute_fold(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any],
    ) -> np.ndarray:
        """Get feature importances from trained model.

        Args:
            model: Trained model
            X: Validation features (unused, kept for interface)
            y: Validation targets (unused, kept for interface)
            model_params: Model hyperparameters (unused)

        Returns:
            Feature importances from model
        """
        importances = model.get_feature_importances()
        if importances is not None:
            return importances
        raise ValueError(
            f"Model {type(model).__name__} does not have feature importances "
            "(no feature_importances_ or coef_ attributes)"
        )
