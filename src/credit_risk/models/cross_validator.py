"""Cross-validation with injected splitter, metrics, and model factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from credit_risk.models.metrics import ClassificationMetrics, ClassificationRankingMetrics
from credit_risk.models.model_factory import ModelFactory
from credit_risk.models.splitter import Splitter

if TYPE_CHECKING:
    from credit_risk.models.importance.base import BaseImportanceStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "CVMetrics",
    "CVResult",
    "CVScores",
    "CrossValidator",
    "FoldResult",
]


@dataclass
class FoldResult:
    """Results from a single fold."""

    fold_index: int
    y_true: np.ndarray
    y_prob: np.ndarray
    importances: np.ndarray | None = None


@dataclass
class CVResult:
    """Results from cross-validation run - raw fold data."""

    n_folds: int
    n_features: int
    fold_results: list[FoldResult]
    y_true: np.ndarray | None = None
    y_prob: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.y_true is None and self.fold_results:
            self.y_true = np.concatenate([fr.y_true for fr in self.fold_results])
            self.y_prob = np.concatenate([fr.y_prob for fr in self.fold_results])


@dataclass
class CVScores:
    """Computed metrics from CVResult."""

    mean_scores: dict[str, float]
    std_scores: dict[str, float]
    fold_scores: list[dict[str, float]]
    mean_importances: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result: dict[str, Any] = {}
        for metric, mean_value in self.mean_scores.items():
            result[f"mean_{metric}"] = mean_value
            result[f"std_{metric}"] = self.std_scores.get(metric, 0)
        if self.mean_importances is not None:
            result["mean_importances"] = self.mean_importances.tolist()
        return result


class CVMetrics:
    """Compute metrics from CVResult."""

    @staticmethod
    def compute(
        cv_result: CVResult,
        metrics: ClassificationRankingMetrics | ClassificationMetrics,
    ) -> CVScores:
        """Compute aggregated metrics from CVResult.

        Args:
            cv_result: Cross-validation result with fold data
            metrics: Metrics to compute per fold

        Returns:
            CVScores with mean/std metrics
        """
        if not cv_result.fold_results:
            return CVScores(mean_scores={}, std_scores={}, fold_scores=[])

        fold_scores: list[dict[str, float]] = []
        for fold_fr in cv_result.fold_results:
            fold_scores.append(metrics.compute(fold_fr.y_true, fold_fr.y_prob))

        metrics_list = list(fold_scores[0].keys())
        mean_scores: dict[str, float] = {}
        std_scores: dict[str, float] = {}

        for metric in metrics_list:
            values = [fs[metric] for fs in fold_scores]
            mean_scores[metric] = float(np.mean(values))
            std_scores[metric] = float(np.std(values))

        mean_importances: np.ndarray | None = None
        importances = [
            fr.importances for fr in cv_result.fold_results if fr.importances is not None
        ]
        if importances:
            mean_importances = np.mean(importances, axis=0)

        return CVScores(
            mean_scores=mean_scores,
            std_scores=std_scores,
            fold_scores=fold_scores,
            mean_importances=mean_importances,
        )


class CrossValidator:
    """Cross-validator with injected splitter and model factory.

    Usage:
        from credit_risk.config import load_config
        cfg = load_config()
        from credit_risk.models.splitter import TrainTestCVSplitter

        splitter = TrainTestCVSplitter(
            n_splits=cfg.splitter.n_splits,
            cv_random_state=cfg.splitter.cv_random_state,
            test_random_state=cfg.splitter.test_random_state,
            stratify=cfg.splitter.stratify,
            shuffle=cfg.splitter.shuffle,
        )
        metrics = ClassificationRankingMetrics()

        validator = CrossValidator(
            splitter=splitter,
            model_factory=factory,
        )
        result = validator.validate(X, y, model_params={...})
        cv_scores = CVMetrics.compute(result, metrics)

    With importance strategy:
        from credit_risk.models.importance import InnerImportance

        validator = CrossValidator(
            splitter=splitter,
            model_factory=factory,
            importance_strategy=InnerImportance(),
        )
        result = validator.validate(X, y, model_params={...})
        cv_scores = CVMetrics.compute(result, metrics)
        importances = cv_scores.mean_importances
    """

    def __init__(
        self,
        splitter: Splitter,
        model_factory: ModelFactory,
        importance_strategy: "BaseImportanceStrategy | None" = None,
        verbose: bool = True,
    ):
        """Initialize CrossValidator.

        Args:
            splitter: Train/test splitter (e.g., StratifiedKFold)
            model_factory: Factory for creating models
            importance_strategy: Optional strategy to compute feature importance
                per fold. If provided, mean_importances will be computed.
            verbose: If True, log progress
        """
        self._splitter = splitter
        self._model_factory = model_factory
        self._importance_strategy = importance_strategy
        self.verbose = verbose
        self._n_splits = getattr(splitter, "n_splits", 5)

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any] | None = None,
    ) -> CVResult:
        """Run cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            model_params: Parameters for model instantiation

        Returns:
            CVResult with raw fold data
        """
        if model_params is None:
            model_params = {}

        fold_results: list[FoldResult] = []

        if callable(getattr(self._splitter, "split_cv", None)):
            cv_splits = self._splitter.split_cv(X, y)  # type: ignore[union-attr]
        else:
            cv_splits = self._splitter.split(X, y)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            if self.verbose:
                logger.info(f"Fold {fold_idx + 1}/{self._n_splits}...")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = self._model_factory(**model_params).build_model_pipeline()
            model.fit(X_train_fold, y_train_fold)

            y_pred_proba = model.predict_proba(X_val_fold)

            fold_imp: np.ndarray | None = None
            if self._importance_strategy:
                fold_imp = self._importance_strategy.compute_fold(
                    model, X_val_fold, y_val_fold, model_params
                )

            fold_results.append(
                FoldResult(
                    fold_index=fold_idx,
                    y_true=y_val_fold,
                    y_prob=y_pred_proba,
                    importances=fold_imp,
                )
            )

            if self.verbose:
                logger.info(f"Fold {fold_idx + 1} done.")

        result = CVResult(
            n_folds=self._n_splits,
            n_features=X.shape[1],
            fold_results=fold_results,
        )

        if self.verbose:
            logger.info(f"CV Results: {result.n_folds} folds, {X.shape[1]} features")

        return result
