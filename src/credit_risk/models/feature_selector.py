"""Backward feature selection driven by ProcessingCV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from credit_risk.models.cross_validator import CVMetrics, CVResult
from credit_risk.models.metrics import ClassificationRankingMetrics

if TYPE_CHECKING:
    from credit_risk.pipeline.cv_pipeline import ProcessingCV

logger = logging.getLogger(__name__)


@dataclass
class EliminationStep:
    """Record of one elimination step."""

    step: int
    n_features: int
    removed_features: list[str]
    cv_result: CVResult


class BackwardFeatureSelector:
    """Backward feature selection using ProcessingCV for zero-leakage evaluation.

    At each elimination step a single ProcessingCV.validate() call is made with
    a feature_mask restricted to the current feature subset.  This gives both
    the CV score (stop decision) and the mean feature importances (removal
    decision) in one pass — no double processing per step.

    Importance strategy must be set on the injected ProcessingCV instance.
    If no importance strategy is set, features are removed in arbitrary order
    (equivalent to random elimination — useful only for ablation studies).

    Args:
        processing_cv: Fully configured ProcessingCV instance (includes
            TableTransformer, splitter, model factory, importance strategy).
        metrics: Metrics to compute per fold for stop decisions.
        selection_metric_name: Metric used for the stop criterion.
        min_features: Stop eliminating when this many features remain.
        tolerance: Maximum allowed drop in selection_metric before stopping.
        nb_remove_features: Number (int) or fraction (float < 1) of features
            to remove per step.
        verbose: Log progress at each step.

    Usage:
        processing_cv = ProcessingCV(
            table_transformer=tt,
            splitter=splitter,
            model_factory=factory,
            importance_strategy=InnerImportance(),
        )
        selector = BackwardFeatureSelector(processing_cv=processing_cv, ...)
        best_features, best_cv_result = selector.eliminate(
            tables=raw_tables,
            labels=labels_train_df,
            model_params=cfg.model.params,
        )
    """

    def __init__(
        self,
        processing_cv: "ProcessingCV",
        metrics: ClassificationRankingMetrics,
        selection_metric_name: str = "roc_auc",
        min_features: int = 5,
        tolerance: float = 0.005,
        nb_remove_features: int | float = 1,
        verbose: bool = True,
    ) -> None:
        self._processing_cv = processing_cv
        self._metrics = metrics
        self._selection_metric_name = selection_metric_name
        self.min_features = min_features
        self.tolerance = tolerance
        self.nb_remove_features = nb_remove_features
        self.verbose = verbose
        self._n_splits = getattr(processing_cv.splitter, "n_splits", 5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eliminate(
        self,
        tables: dict[str, Any],
        labels: Any,
        model_params: dict[str, Any] | None = None,
    ) -> tuple[list[str], CVResult]:
        """Run backward feature elimination.

        Args:
            tables: Mapping of table name → raw Polars DataFrame.
            labels: Polars DataFrame with id and target columns (train only).
            model_params: Hyperparameters passed to the model factory.

        Returns:
            Tuple of (best_feature_names, best_cv_result) where both
            correspond to the step with the highest CV score.
        """
        model_params = model_params or {}

        if self.verbose:
            logger.info("Starting backward feature selection")

        # ── Initial pass: full feature set ───────────────────────────────────
        initial_result = self._processing_cv.validate(tables, labels, model_params)
        feature_names: list[str] = initial_result.feature_names

        if not feature_names:
            raise RuntimeError(
                "ProcessingCV returned no feature names. "
                "Ensure TableTransformer produces at least one feature column."
            )

        initial_scores = CVMetrics.compute(initial_result, self._metrics)
        best_score = initial_scores.mean_scores[self._selection_metric_name]
        # Importances from initial pass drive the first removal decision.
        next_importances = initial_scores.mean_importances

        if self.verbose:
            logger.info(f"Starting: {len(feature_names)} features, {self._n_splits}-fold CV")
            logger.info(
                f"Step 1: {len(feature_names)} features, "
                f"{self._selection_metric_name}: {best_score:.4f}"
            )

        step = 1
        history: list[EliminationStep] = [
            EliminationStep(
                step=step,
                n_features=len(feature_names),
                removed_features=[],
                cv_result=initial_result,
            )
        ]

        current_features = list(feature_names)

        # ── Elimination loop ─────────────────────────────────────────────────
        while len(current_features) > self.min_features:
            n_to_remove = self._get_nb_to_remove(len(current_features))
            n_to_remove = min(n_to_remove, len(current_features) - self.min_features)

            # Decide which features to remove using importances from previous pass.
            if next_importances is not None and next_importances.shape[0] == len(current_features):
                importances = next_importances
            else:
                importances = np.zeros(len(current_features))

            least_important_idx = np.argsort(importances)[:n_to_remove]
            removed = [current_features[i] for i in least_important_idx]
            remaining = [f for i, f in enumerate(current_features) if i not in least_important_idx]

            # Combined pass: score + importances on the remaining feature set.
            result = self._processing_cv.validate(
                tables, labels, model_params, feature_mask=remaining
            )
            scores = CVMetrics.compute(result, self._metrics)
            new_score = scores.mean_scores[self._selection_metric_name]
            delta = new_score - best_score
            next_importances = scores.mean_importances  # used in next iteration

            step += 1

            removed_str = ", ".join(removed[:3])
            if len(removed) > 3:
                removed_str += f"... (+{len(removed) - 3} more)"

            if self.verbose:
                logger.info(
                    f"Step {step}: Remove [{removed_str}] "
                    f"({len(current_features)}->{len(remaining)}), "
                    f"{self._selection_metric_name}: {new_score:.4f} (d: {delta:+.4f})"
                )

            if new_score >= best_score - self.tolerance:
                current_features = remaining
                history.append(
                    EliminationStep(
                        step=step,
                        n_features=len(current_features),
                        removed_features=removed,
                        cv_result=result,
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

        # ── Find best step ───────────────────────────────────────────────────
        best_step = max(
            history,
            key=lambda s: CVMetrics.compute(s.cv_result, self._metrics).mean_scores[
                self._selection_metric_name
            ],
        )

        # Reconstruct best feature names by replaying removals up to best_step.
        if best_step.step == 1:
            best_features = list(feature_names)
        else:
            remaining = list(feature_names)
            for h in history[1:]:
                if h.step <= best_step.step and h.removed_features:
                    for f in h.removed_features:
                        remaining = [r for r in remaining if r != f]
            best_features = remaining

        if self.verbose:
            best_scores = CVMetrics.compute(best_step.cv_result, self._metrics)
            best_auc = best_scores.mean_scores.get(self._selection_metric_name, 0)
            best_std = best_scores.std_scores.get(self._selection_metric_name, 0)
            logger.info(f"BEST: {best_step.n_features} features")
            logger.info(f"  CV {self._selection_metric_name}: {best_auc:.4f} +/- {best_std:.4f}")
            final_scores = CVMetrics.compute(history[-1].cv_result, self._metrics)
            final_auc = final_scores.mean_scores.get(self._selection_metric_name, 0)
            final_std = final_scores.std_scores.get(self._selection_metric_name, 0)
            logger.info(
                f"Final: {history[-1].n_features} features — "
                f"CV {self._selection_metric_name}: {final_auc:.4f} +/- {final_std:.4f}"
            )

        return best_features, best_step.cv_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_nb_to_remove(self, n_features_left: int) -> int:
        if self.nb_remove_features >= 1:
            return int(self.nb_remove_features)
        return max(1, int(self.nb_remove_features * n_features_left))
