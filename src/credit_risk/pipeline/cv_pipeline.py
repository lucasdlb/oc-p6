"""Zero-leakage CV orchestrator.

ProcessingCV is a thin orchestrator that:
  1. Splits data into folds
  2. Delegates table processing to TableTransformer (fit/transform, join, numpy)
  3. Trains model per fold
  4. Collects fold results for CV metrics

ProcessingCV knows nothing about Polars, pipelines, or table processing — those
are handled by TableTransformer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from credit_risk.models.cross_validator import CVResult, FoldResult
from credit_risk.models.model_factory import ModelFactory

if TYPE_CHECKING:
    from credit_risk.models.importance.base import BaseImportanceStrategy
    from credit_risk.pipeline.table_transformer import TableTransformer

logger = logging.getLogger(__name__)


@dataclass
class FoldArrays:
    """Pre-built numpy arrays for one CV fold."""

    fold_index: int
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]


class ProcessingCV:
    """Zero-leakage CV orchestrator.

    Per fold:
      1. TableTransformer.fit_transform(tables, labels, train_ids, val_ids) → X/y numpy
      2. Model fit/predict
      3. Collect FoldResult
    """

    def __init__(
        self,
        table_transformer: "TableTransformer",
        splitter,
        model_factory: ModelFactory,
        importance_strategy: "BaseImportanceStrategy | None" = None,
        verbose: bool = True,
    ) -> None:
        self.table_transformer = table_transformer
        self.splitter = splitter
        self.model_factory = model_factory
        self.importance_strategy = importance_strategy
        self.verbose = verbose

    def build_folds(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        feature_mask: list[str] | None = None,
    ) -> list[FoldArrays]:
        """Pre-build all fold arrays once — process tables per fold, return numpy.

        This separates the expensive table processing from the model fitting so
        that Optuna parallel workers can share the pre-built arrays without
        reprocessing tables in every trial.

        Args:
            tables: Raw Polars DataFrames.
            labels: Labels DataFrame (train IDs only).
            feature_mask: Optional feature subset to apply after processing.

        Returns:
            List of FoldArrays, one per CV fold.
        """
        ids = labels.select(self.table_transformer.id_column).to_numpy().ravel()
        y = labels.select(self.table_transformer.target_column).to_numpy().ravel()

        folds: list[FoldArrays] = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(self.splitter.split_cv(ids, y)):
            if self.verbose:
                logger.info(
                    "\n%s\n Preprocessing fold %d/%d\n%s",
                    "─" * 50,
                    fold_idx + 1,
                    self.splitter.n_splits,
                    "─" * 50,
                )

            fold_train_ids = set(ids[fold_train_idx].tolist())
            fold_val_ids = set(ids[fold_val_idx].tolist())

            X_train, X_val, y_train, y_val, feature_names = self.table_transformer.fit_transform(
                tables, labels, fold_train_ids, fold_val_ids
            )

            if feature_mask is not None:
                available = set(feature_names)
                fold_mask = [f for f in feature_mask if f in available]
                mask_idx = [feature_names.index(f) for f in fold_mask]
                X_train = X_train[:, mask_idx]
                X_val = X_val[:, mask_idx]
                feature_names = fold_mask

            if self.verbose:
                logger.info(
                    "  Fold %d: X_train=%s  X_val=%s  pos_rate=%.3f",
                    fold_idx + 1,
                    X_train.shape,
                    X_val.shape,
                    y_train.mean(),
                )

            folds.append(
                FoldArrays(
                    fold_index=fold_idx,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    feature_names=feature_names,
                )
            )

        return folds

    def validate_folds(
        self,
        folds: list[FoldArrays],
        model_params: dict[str, Any] | None = None,
    ) -> CVResult:
        """Run CV on pre-built fold arrays — no table processing, model only.

        Used by ProcessingTuner when fold arrays are pre-built once and reused
        across all Optuna trials.

        Args:
            folds: Pre-built fold arrays from build_folds().
            model_params: Hyperparameters for the model pipeline.

        Returns:
            CVResult with fold_results and feature_names populated.
        """
        model_params = model_params or {}
        fold_results: list[FoldResult] = []

        for fa in folds:
            model = self.model_factory(**model_params).build_model_pipeline()
            model.fit(fa.X_train, fa.y_train)
            y_pred = model.predict_proba(fa.X_val)

            importances: np.ndarray | None = None
            if self.importance_strategy is not None:
                importances = self.importance_strategy.compute_fold(
                    model, fa.X_val, fa.y_val, model_params
                )

            fold_results.append(
                FoldResult(
                    fold_index=fa.fold_index,
                    y_true=fa.y_val,
                    y_prob=y_pred,
                    importances=importances,
                    feature_names=fa.feature_names,
                )
            )

        return CVResult(
            n_folds=self.splitter.n_splits,
            n_features=folds[0].X_train.shape[1] if folds else 0,
            fold_results=fold_results,
            feature_names=folds[0].feature_names if folds else [],
        )

    def validate(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_params: dict[str, Any] | None = None,
        feature_mask: list[str] | None = None,
    ) -> CVResult:
        """Run zero-leakage cross-validation.

        Args:
            tables: Mapping of table name → raw Polars DataFrame.
            labels: Polars DataFrame with id_column and target_column columns.
            model_params: Hyperparameters for the model pipeline.
            feature_mask: Optional list of feature names to restrict to after
                processing.  When set, only these features are passed to the
                model; importances are aligned to this subset.  Useful for
                BackwardFeatureSelector to evaluate a feature subset without
                re-processing raw tables from scratch.

        Returns:
            CVResult with fold_results and feature_names populated.
        """
        model_params = model_params or {}

        ids = labels.select(self.table_transformer.id_column).to_numpy().ravel()
        y = labels.select(self.table_transformer.target_column).to_numpy().ravel()

        n_splits = self.splitter.n_splits
        fold_results: list[FoldResult] = []
        active_feature_names: list[str] = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(self.splitter.split_cv(ids, y)):
            if self.verbose:
                logger.info(f"\n{'─' * 50}\n Fold {fold_idx + 1}/{n_splits}\n{'─' * 50}")

            fold_train_ids = set(ids[fold_train_idx].tolist())
            fold_val_ids = set(ids[fold_val_idx].tolist())

            X_train, X_val, y_train, y_val, feature_names = self.table_transformer.fit_transform(
                tables, labels, fold_train_ids, fold_val_ids
            )

            # Apply feature mask: restrict to the requested subset.
            # Features in the mask that were dropped by the encoder in this
            # fold (e.g. high-cardinality columns near max_categories boundary)
            # are silently skipped — the model sees a slightly smaller set for
            # that fold, which is acceptable.
            if feature_mask is not None:
                available = set(feature_names)
                fold_mask = [f for f in feature_mask if f in available]
                mask_idx = [feature_names.index(f) for f in fold_mask]
                X_train = X_train[:, mask_idx]
                X_val = X_val[:, mask_idx]
                active_feature_names = fold_mask
            else:
                active_feature_names = feature_names

            if self.verbose:
                pos_rate = y_train.mean()
                logger.info(
                    f"  X_train: {X_train.shape}  X_val: {X_val.shape}  "
                    f"pos_rate_train: {pos_rate:.3f}"
                )

            model = self.model_factory(**model_params).build_model_pipeline()
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)

            importances: np.ndarray | None = None
            if self.importance_strategy is not None:
                importances = self.importance_strategy.compute_fold(
                    model, X_val, y_val, model_params
                )

            fold_results.append(
                FoldResult(
                    fold_index=fold_idx,
                    y_true=y_val,
                    y_prob=y_pred,
                    importances=importances,
                    feature_names=active_feature_names,
                )
            )

        return CVResult(
            n_folds=n_splits,
            n_features=X_train.shape[1] if fold_results else 0,
            fold_results=fold_results,
            feature_names=active_feature_names,
        )
