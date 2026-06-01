"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    def __init__(self):
        self.metrics: dict[str, float] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> dict[str, float]:
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "average_precision": average_precision_score(y_true, y_pred_proba),
        }
        return self.metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def get_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
        return roc_curve(y_true, y_pred_proba)

    def get_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return auc(fpr, tpr)

    def gini_coefficient(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return 2 * roc_auc_score(y_true, y_pred_proba) - 1
