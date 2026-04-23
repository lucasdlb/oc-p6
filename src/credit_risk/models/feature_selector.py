"""Feature selection with swappable importance strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from credit_risk.models.cross_validator import (
    CrossValidator,
    CVMetrics,
    CVResult,
)
from credit_risk.models.importance.base import BaseImportanceStrategy
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.model_factory import EstimatorPipelineFactory
from credit_risk.models.splitter import Splitter

logger = logging.getLogger(__name__)


@dataclass
class EliminationStep:
    """Record of one elimination step."""

    step: int
    n_features: int
    removed_features: list[str]
    cv_result: CVResult


class BackwardFeatureSelector:
    """Backward feature selection using cross-validated performance.

    Repeatedly removes the least important feature and re-evaluates using
    stratified k-fold cross-validation to get robust performance estimates.
    Stops when performance drops significantly or minimum features reached.

    Uses pluggable importance strategies via FeatureImportance protocol.

    Usage:
        from sklearn.model_selection import StratifiedKFold
        from credit_risk.models import (
            BackwardFeatureSelector,
            LGBMImportance,
            ClassificationRankingMetrics,
            LGBMFactory,
        )

        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics = ClassificationRankingMetrics()
        factory = LGBMFactory()
        importance_strategy = LGBMImportance(splitter, metrics, factory)

        selector = BackwardFeatureSelector(
            splitter=splitter,
            metrics=metrics,
            model_factory=factory,
            importance_strategy=importance_strategy,
            selection_metric_name="roc_auc",
            min_features=5,
            tolerance=0.005,
        )
        best_features, result = selector.eliminate(X, y, model_params={...})
    """

    def __init__(
        self,
        splitter: Splitter,
        metrics: ClassificationRankingMetrics,
        model_factory: EstimatorPipelineFactory,
        importance_strategy: BaseImportanceStrategy | None = None,
        selection_metric_name: str = "roc_auc",
        min_features: int = 5,
        tolerance: float = 0.005,
        nb_remove_features: int | float = 1,
        verbose: bool = True,
    ):
        """Initialize BackwardFeatureSelector."""
        self._splitter = splitter
        self._metrics = metrics
        self._model_factory = model_factory
        self._importance_strategy = importance_strategy
        self._selection_metric_name = selection_metric_name
        self.min_features = min_features
        self.tolerance = tolerance
        self.nb_remove_features = nb_remove_features
        self.verbose = verbose
        self._n_splits = getattr(splitter, "n_splits", 5)

    def _get_nb_to_remove(self, n_features_left: int) -> int:
        if self.nb_remove_features >= 1:
            return int(self.nb_remove_features)
        return max(1, int(self.nb_remove_features * n_features_left))

    def eliminate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
    ) -> tuple[list[str], CVResult]:
        """Run backward feature selection."""
        if model_params is None:
            model_params = {}

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        current_feature_names = list(feature_names)
        current_feature_indices = list(range(X.shape[1]))
        step = 0
        history: list[EliminationStep] = []

        validator = CrossValidator(
            splitter=self._splitter,
            model_factory=self._model_factory,
            verbose=False,
        )

        if self.verbose:
            logger.info("Starting backward feature selection")
            logger.info(f"Starting: {X.shape[1]} features, {self._n_splits}-fold CV")

        initial_result = validator.validate(X, y, model_params)
        initial_scores = CVMetrics.compute(initial_result, self._metrics)
        best_score = initial_scores.mean_scores[self._selection_metric_name]

        step += 1
        history.append(
            EliminationStep(
                step=step,
                n_features=len(current_feature_names),
                removed_features=[],
                cv_result=initial_result,
            )
        )

        if self.verbose:
            logger.info(
                f"Step {step}: {len(current_feature_names)} features, "
                f"{self._selection_metric_name}: {best_score:.4f}"
            )

        while len(current_feature_names) > self.min_features:
            n_to_remove = self._get_nb_to_remove(len(current_feature_names))
            n_to_remove = min(n_to_remove, len(current_feature_names) - self.min_features)

            X_current = X[:, current_feature_indices]

            # Delegate importance computation to CrossValidator with strategy
            if self._importance_strategy:
                validator_imp = CrossValidator(
                    splitter=self._splitter,
                    model_factory=self._model_factory,
                    importance_strategy=self._importance_strategy,
                    verbose=False,
                )
                result_imp = validator_imp.validate(X_current, y, model_params)
                imp_scores = CVMetrics.compute(result_imp, self._metrics)
                imp = imp_scores.mean_importances
                if imp is None or imp.shape[0] != len(current_feature_names):
                    importances = np.zeros(X_current.shape[1])
                else:
                    importances = imp
            else:
                importances = np.zeros(X_current.shape[1])

            least_important_indices = np.argsort(importances)[:n_to_remove]
            removed_feature_names = [current_feature_names[i] for i in least_important_indices]
            removed_feature_indices = [current_feature_indices[i] for i in least_important_indices]

            remaining_feature_names = [
                f for i, f in enumerate(current_feature_names) if i not in least_important_indices
            ]
            remaining_feature_indices = [
                i for i in current_feature_indices if i not in removed_feature_indices
            ]

            if remaining_feature_names:
                X_remaining = X[:, remaining_feature_indices]
                validator_rem = CrossValidator(
                    splitter=self._splitter,
                    model_factory=self._model_factory,
                    verbose=False,
                )
                new_result = validator_rem.validate(X_remaining, y, model_params)
            else:
                new_result = initial_result

            new_scores = CVMetrics.compute(new_result, self._metrics)
            new_score = new_scores.mean_scores[self._selection_metric_name]
            delta = new_score - best_score

            step += 1

            removed_str = ", ".join(removed_feature_names[:3])
            if len(removed_feature_names) > 3:
                removed_str += f"... (+{len(removed_feature_names) - 3} more)"

            if self.verbose:
                old_n = len(current_feature_names)
                new_n = len(remaining_feature_names)
                logger.info(
                    f"Step {step}: Remove [{removed_str}] ({old_n}->{new_n}), "
                    f"{self._selection_metric_name}: {new_score:.4f} (d: {delta:+.4f})"
                )

            if new_score >= best_score - self.tolerance:
                current_feature_names = remaining_feature_names
                current_feature_indices = remaining_feature_indices
                history.append(
                    EliminationStep(
                        step=step,
                        n_features=len(current_feature_names),
                        removed_features=removed_feature_names,
                        cv_result=new_result,
                    )
                )
                if new_score > best_score:
                    best_score = new_score
            else:
                if self.verbose:
                    logger.info(
                        f"Stopping: {self._selection_metric_name} dropped by "
                        f"{-delta:.4f} > tolerance {self.tolerance}"
                    )
                break

        best_step = max(
            history,
            key=lambda s: CVMetrics.compute(s.cv_result, self._metrics).mean_scores[
                self._selection_metric_name
            ],
        )

        best_feature_names = []
        if best_step.step == 1:
            best_feature_names = list(feature_names)
        else:
            remaining = list(feature_names)
            for h in history[1:]:
                if h.step <= best_step.step and h.removed_features:
                    for f in h.removed_features:
                        remaining = [r for r in remaining if r != f]
            best_feature_names = remaining

        if self.verbose:
            best_scores = CVMetrics.compute(best_step.cv_result, self._metrics)
            logger.info(f"BEST: {best_step.n_features} features")
            best_auc_val = best_scores.mean_scores.get(self._selection_metric_name, 0)
            best_std = best_scores.std_scores.get(self._selection_metric_name, 0)
            logger.info(
                f"  CV {self._selection_metric_name}: {best_auc_val:.4f} +/- {best_std:.4f}"
            )
            logger.info(f"Final: {len(current_feature_names)} features (stopped at min_features)")
            final_scores = CVMetrics.compute(history[-1].cv_result, self._metrics)
            final_auc = final_scores.mean_scores.get(self._selection_metric_name, 0)
            final_std = final_scores.std_scores.get(self._selection_metric_name, 0)
            logger.info(f"  CV {self._selection_metric_name}: {final_auc:.4f} +/- {final_std:.4f}")

        return best_feature_names, best_step.cv_result


class BackwardFeatureEliminator(BackwardFeatureSelector):
    """Backward feature elimination using cross-validated performance."""

    def __init__(
        self,
        splitter: Splitter,
        metrics: ClassificationRankingMetrics,
        model_factory: EstimatorPipelineFactory,
        selection_metric_name: str = "roc_auc",
        min_features: int = 5,
        tolerance: float = 0.005,
        nb_remove_features: int | float = 1,
        verbose: bool = True,
    ):
        super().__init__(
            splitter=splitter,
            metrics=metrics,
            model_factory=model_factory,
            importance_strategy=None,
            selection_metric_name=selection_metric_name,
            min_features=min_features,
            tolerance=tolerance,
            nb_remove_features=nb_remove_features,
            verbose=verbose,
        )
