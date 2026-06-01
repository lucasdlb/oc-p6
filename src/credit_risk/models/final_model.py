"""Final model trainer with evaluation on held-out test set."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from credit_risk.models.model_factory import EstimatorPipelineFactory

if TYPE_CHECKING:
    from credit_risk.models.importance.base import BaseImportanceStrategy

logger = logging.getLogger(__name__)


@dataclass
class FinalModelResult:
    """Results from final model training and evaluation."""

    test_roc_auc: float
    test_f1: float
    test_recall: float
    test_precision: float
    test_accuracy: float
    optimal_threshold: float
    feature_importance: list[tuple[str, float]]
    n_features: int
    n_train_samples: int
    n_test_samples: int


class FinalModelTrainer:
    """Train final model and evaluate on held-out test set.

    Usage:
        trainer = FinalModelTrainer(model_factory=model_factory)
        result = trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test,
            best_features, model_params, optimal_threshold=0.5
        )
    """

    def __init__(
        self,
        model_factory: EstimatorPipelineFactory,
        importance_strategy: "BaseImportanceStrategy | None" = None,
    ):
        """Initialize FinalModelTrainer.

        Args:
            model_factory: Factory for creating models
            importance_strategy: Optional strategy to compute feature importance
        """
        self._model_factory = model_factory
        self._importance_strategy = importance_strategy

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        best_features: list[str],
        model_params: dict[str, Any],
        optimal_threshold: float = 0.5,
    ) -> FinalModelResult:
        """Train final model on full training set and evaluate on test set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            best_features: List of selected feature names
            model_params: Model hyperparameters
            optimal_threshold: Threshold for binary prediction

        Returns:
            FinalModelResult with metrics and feature importance
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        logger.info(f"Training final model on training set with {len(best_features)} features")
        logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        model = self._model_factory(**model_params).build_model_pipeline()
        model.fit(X_train, y_train)

        logger.info("Evaluating on held-out TEST SET")
        y_test_proba = model.predict_proba(X_test)
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        logger.info(f"Optimal threshold used: {optimal_threshold:.2f}")
        logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")
        logger.info(f"Test F1: {test_f1:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        if self._importance_strategy is not None:
            importance_values = self._importance_strategy.compute_fold(
                model, X_train, y_train, model_params
            )
        else:
            importance_values = model.get_feature_importances()

        top_features = sorted(
            zip(best_features, importance_values, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info("Feature importance (best set):")
        for feat, imp in top_features:
            logger.info(f"  {feat}: {imp}")

        return FinalModelResult(
            test_roc_auc=test_roc_auc,
            test_f1=test_f1,
            test_recall=test_recall,
            test_precision=test_precision,
            test_accuracy=test_accuracy,
            optimal_threshold=optimal_threshold,
            feature_importance=top_features,
            n_features=len(best_features),
            n_train_samples=X_train.shape[0],
            n_test_samples=X_test.shape[0],
        )
