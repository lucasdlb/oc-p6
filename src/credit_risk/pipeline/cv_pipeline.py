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
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from credit_risk.models.cross_validator import CVResult, FoldResult
from credit_risk.models.model_factory import ModelFactory

if TYPE_CHECKING:
    from credit_risk.models.importance.base import BaseImportanceStrategy
    from credit_risk.pipeline.table_transformer import TableTransformer

logger = logging.getLogger(__name__)


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

    def validate(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_params: dict[str, Any] | None = None,
    ) -> CVResult:
        """Run zero-leakage cross-validation.

        Args:
            tables: Mapping of table name → raw Polars DataFrame.
            labels: Polars DataFrame with id_column and target_column columns.
            model_params: Hyperparameters for the model pipeline.

        Returns:
            CVResult with fold_results populated.
        """
        model_params = model_params or {}

        ids = labels.select(self.table_transformer.id_column).to_numpy().ravel()
        y = labels.select(self.table_transformer.target_column).to_numpy().ravel()

        n_splits = self.splitter.n_splits
        fold_results: list[FoldResult] = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(self.splitter.split_cv(ids, y)):
            if self.verbose:
                logger.info(f"\n{'─' * 50}\n Fold {fold_idx + 1}/{n_splits}\n{'─' * 50}")

            fold_train_ids = set(ids[fold_train_idx].tolist())
            fold_val_ids = set(ids[fold_val_idx].tolist())

            X_train, X_val, y_train, y_val, feature_names = self.table_transformer.fit_transform(
                tables, labels, fold_train_ids, fold_val_ids
            )

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
                )
            )

        return CVResult(
            n_folds=n_splits,
            n_features=X_train.shape[1] if fold_results else 0,
            fold_results=fold_results,
        )
