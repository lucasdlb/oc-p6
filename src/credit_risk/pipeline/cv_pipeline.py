"""Zero-leakage CV orchestrator for raw Polars data.

ProcessingCV fits a fresh ProcessingPipeline per table per fold, ensuring
imputer/encoder statistics never see validation data.

ProcessingCV is config-agnostic — it knows nothing about TableConfig or how
to build ProcessingPipeline. The caller supplies factories that create a
fresh pipeline per table per fold.

Usage:
    from credit_risk.models.splitter import TrainTestCVSplitter

    cv = ProcessingCV(
        pipeline_factories={
            "application": lambda: ProcessingPipeline(cfg.data.application),
        },
        splitter=TrainTestCVSplitter(n_splits=5, cv_random_state=42),
        model_factory=get_factory(cfg.model.model_type, cfg.model.x_transform),
    )
    result = cv.validate(
        tables={"application": df_raw},
        labels=labels_df,          # SK_ID_CURR + TARGET columns
        model_params=cfg.model.params,
    )
    scores = CVMetrics.compute(result, ClassificationRankingMetrics())
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from credit_risk.models.cross_validator import CVResult, FoldResult
from credit_risk.models.model_factory import ModelFactory
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

if TYPE_CHECKING:
    from credit_risk.models.importance.base import BaseImportanceStrategy

logger = logging.getLogger(__name__)


class ProcessingCV:
    """Zero-leakage CV orchestrator.

    Per fold:
      1. Split applicant IDs into train / val sets
      2. Per table: fresh ProcessingPipeline — fit(train only) → transform(train + val)
      3. Prefix all columns (except id_column and target_column) with table name
      4. Left-join processed tables onto labels
      5. Align columns between train and val
      6. Convert to numpy → model.fit() → model.predict_proba()
    """

    def __init__(
        self,
        pipeline_factories: dict[str, Callable[[], ProcessingPipeline]],
        splitter,
        model_factory: ModelFactory,
        importance_strategy: "BaseImportanceStrategy | None" = None,
        id_column: str = "SK_ID_CURR",
        target_column: str = "TARGET",
        verbose: bool = True,
    ) -> None:
        self.pipeline_factories = pipeline_factories
        self.splitter = splitter
        self.model_factory = model_factory
        self.importance_strategy = importance_strategy
        self.id_column = id_column
        self.target_column = target_column
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
                    Must contain the id_column (no labels joined yet).
            labels: Polars DataFrame with id_column and target_column columns.
            model_params: Hyperparameters for the model pipeline.

        Returns:
            CVResult with fold_results populated.
        """
        model_params = model_params or {}

        ids = labels.select(self.id_column).to_numpy().ravel()
        y = labels.select(self.target_column).to_numpy().ravel()

        n_splits = self.splitter.n_splits
        fold_results: list[FoldResult] = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(self.splitter.split_cv(ids, y)):
            if self.verbose:
                logger.info(f"\n{'─' * 50}\n Fold {fold_idx + 1}/{n_splits}\n{'─' * 50}")

            fold_train_ids = set(ids[fold_train_idx].tolist())
            fold_val_ids = set(ids[fold_val_idx].tolist())

            # ── 1. Process each table independently (fresh pipeline per fold) ──
            fold_train_dfs, fold_val_dfs = self._process_tables(
                tables, fold_train_ids, fold_val_ids
            )

            # ── 2. Merge + numpy conversion ───────────────────────────────────
            X_train, X_val, y_train, y_val, feature_names = self._merge_and_convert(
                fold_train_dfs, fold_val_dfs, labels, fold_train_ids, fold_val_ids
            )

            if self.verbose:
                pos_rate = y_train.mean()
                logger.info(
                    f"  X_train: {X_train.shape}  X_val: {X_val.shape}  "
                    f"pos_rate_train: {pos_rate:.3f}"
                )

            # ── 3. Model training + prediction ───────────────────────────────────
            model = self.model_factory(**model_params).build_model_pipeline()
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)

            # ── 4. Feature importances (optional) ───────────────────────────────
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

    # ── Internal methods ─────────────────────────────────────────────────────

    def _process_tables(
        self,
        tables: dict[str, pl.DataFrame],
        train_ids: set,
        val_ids: set,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        """Process all tables: fit on train, transform train + val.

        A fresh ProcessingPipeline is instantiated per table per fold —
        no state leaks between folds.
        """
        train_dfs: dict[str, pl.DataFrame] = {}
        val_dfs: dict[str, pl.DataFrame] = {}

        for name, raw in tables.items():
            # Drop target column if present — pipeline must never see it
            if self.target_column in raw.columns:
                raw = raw.drop(self.target_column)

            df_train_raw = raw.filter(pl.col(self.id_column).is_in(train_ids))
            df_val_raw = raw.filter(pl.col(self.id_column).is_in(val_ids))

            # Fresh pipeline from factory — zero leakage
            pipe = self.pipeline_factories[name]()
            pipe.fit(df_train_raw)

            train_out = pipe.transform(df_train_raw)
            val_out = pipe.transform(df_val_raw)

            train_dfs[name] = self._prefix_columns(train_out, name)
            val_dfs[name] = self._prefix_columns(val_out, name)

            if self.verbose:
                logger.info(
                    f"  [{name}] train={len(df_train_raw):,} rows → {train_dfs[name].shape[1]} cols"
                )

        return train_dfs, val_dfs

    def _prefix_columns(self, df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """Prefix all columns except id_column and target_column with table name."""
        rename_map = {
            col: f"{prefix}_{col}"
            for col in df.columns
            if col != self.id_column and col != self.target_column
        }
        return df.rename(rename_map)

    def _merge_and_convert(
        self,
        train_dfs: dict[str, pl.DataFrame],
        val_dfs: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        train_ids: set,
        val_ids: set,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Left-join all processed tables onto labels, convert to numpy.

        Column alignment: train defines the feature schema. Val columns
        missing in train are added as null. Val columns not in train are dropped.
        No fill_null — nulls become NaN and expose logic errors immediately.
        """
        # ── 1. Build per-fold label anchors ───────────────────────────────────
        labels_train = labels.filter(pl.col(self.id_column).is_in(train_ids))
        labels_val = labels.filter(pl.col(self.id_column).is_in(val_ids))

        # ── 2. Left-join all tables sequentially ───────────────────────────────
        merged_train = self._join_all(labels_train, train_dfs)
        merged_val = self._join_all(labels_val, val_dfs)

        # ── 3. Determine feature columns ──────────────────────────────────────
        drop_cols = {self.id_column, self.target_column}
        feature_cols = [c for c in merged_train.columns if c not in drop_cols]

        # Hard fail — any string/categorical here means the pipeline missed it
        bad_cols = [c for c in feature_cols if merged_train[c].dtype in (pl.Utf8, pl.Categorical)]
        if bad_cols:
            raise ValueError(
                f"Non-numeric columns found after pipeline processing: {bad_cols}. "
                f"Encoding step must handle all categorical columns."
            )

        # ── 4. Align val to train schema ──────────────────────────────────────
        val_feature_cols = {c for c in merged_val.columns if c not in drop_cols}
        missing_in_val = [c for c in feature_cols if c not in val_feature_cols]
        extra_in_val = [c for c in val_feature_cols if c not in feature_cols]

        if missing_in_val or extra_in_val:
            raise ValueError(
                f"Schema mismatch between train and val after pipeline processing.\n"
                f"  Missing in val : {missing_in_val}\n"
                f"  Extra in val   : {extra_in_val}\n"
                f"Encoding/aggregation steps must produce identical schemas on train and val."
            )

        # Ensure same column order
        merged_val = merged_val.select([self.id_column, self.target_column] + feature_cols)

        # ── 5. Convert to numpy ────────────────────────────────────────────────
        y_train = merged_train.select(self.target_column).to_numpy().ravel()
        y_val = merged_val.select(self.target_column).to_numpy().ravel()

        X_train = merged_train.select(feature_cols).to_numpy()
        X_val = merged_val.select(feature_cols).to_numpy()

        return X_train, X_val, y_train, y_val, feature_cols

    def _join_all(
        self,
        base: pl.DataFrame,
        dfs: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Left-join all processed tables onto base (labels) by id_column."""
        result = base
        for _name, df in dfs.items():
            result = result.join(df, on=self.id_column, how="left")
        return result
