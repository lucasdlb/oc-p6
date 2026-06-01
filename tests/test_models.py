"""Tests for model components."""

import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk.models.evaluator import ModelEvaluator


def test_evaluator_calculates_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 1.0


def test_evaluator_gini_coefficient():
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

    evaluator = ModelEvaluator()
    gini = evaluator.gini_coefficient(y_true, y_pred_proba)
    expected_gini = 2 * roc_auc_score(y_true, y_pred_proba) - 1
    assert abs(gini - expected_gini) < 0.01
