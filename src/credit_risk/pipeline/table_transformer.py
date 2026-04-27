"""Raw Polars tables → numpy X, y for model training.

TableTransformer fits a fresh sklearn Pipeline per table per fold (zero
leakage), joins processed tables, and returns numpy arrays ready for
model training.

Usage:
    tt = TableTransformer(
        pipeline_factories={
            "application": lambda: ProcessingPipeline(cfg.data.application).build(),
        },
    )
    X_train, X_val, y_train, y_val, feature_names = tt.fit_transform(
        tables={"application": df_raw},
        labels=labels_df,
        train_ids=train_ids,
        val_ids=val_ids,
    )
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline


class TableTransformer:
    """Transform raw Polars tables into numpy arrays for model training.

    Per call (fit_transform on train_ids):
      1. Filter each table to train_ids / val_ids
      2. Per table: fresh Pipeline from factory → fit(train) → transform(train + val)
      3. Prefix columns with table name
      4. Left-join all tables onto labels
      5. Validate schema alignment, convert to numpy

    No state is kept between calls — fresh pipelines are created each time.
    """

    def __init__(
        self,
        pipeline_factories: dict[str, Callable[[], Pipeline]],
        id_column: str = "SK_ID_CURR",
        target_column: str = "TARGET",
    ) -> None:
        self.pipeline_factories = pipeline_factories
        self.id_column = id_column
        self.target_column = target_column

    def fit_transform(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        train_ids: set,
        val_ids: set,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Transform tables: fit pipelines on train, transform train + val, join, convert to numpy.

        Args:
            tables: Mapping of table name → raw Polars DataFrame.
            labels: Polars DataFrame with id_column and target_column columns.
            train_ids: Set of IDs for training fold.
            val_ids: Set of IDs for validation fold.

        Returns:
            X_train, X_val, y_train, y_val, feature_names
        """
        train_dfs, val_dfs = self._process_tables(tables, train_ids, val_ids)
        return self._merge_and_convert(train_dfs, val_dfs, labels, train_ids, val_ids)

    # ── Internal methods ─────────────────────────────────────────────────────

    def _process_tables(
        self,
        tables: dict[str, pl.DataFrame],
        train_ids: set,
        val_ids: set,
    ) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
        """Process all tables: fit(train) → transform(train + val).

        A fresh Pipeline is created per table via factory — no state leaks.
        """
        train_dfs: dict[str, pl.DataFrame] = {}
        val_dfs: dict[str, pl.DataFrame] = {}

        for name, raw in tables.items():
            df_train_raw = raw.filter(pl.col(self.id_column).is_in(train_ids))
            df_val_raw = raw.filter(pl.col(self.id_column).is_in(val_ids))

            y_train_raw = None
            if self.target_column in df_train_raw.columns:
                y_train_raw = df_train_raw.select(pl.col(self.target_column))
                df_train_raw = df_train_raw.drop(self.target_column)
                df_val_raw = df_val_raw.drop(self.target_column)

            pipe = self.pipeline_factories[name]()
            pipe.fit(df_train_raw, y=y_train_raw)

            train_out = pipe.transform(df_train_raw)
            val_out = pipe.transform(df_val_raw)

            train_dfs[name] = self._prefix_columns(train_out, name)
            val_dfs[name] = self._prefix_columns(val_out, name)

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
        labels_train = labels.filter(pl.col(self.id_column).is_in(train_ids))
        labels_val = labels.filter(pl.col(self.id_column).is_in(val_ids))

        merged_train = self._join_all(labels_train, train_dfs)
        merged_val = self._join_all(labels_val, val_dfs)

        drop_cols = {self.id_column, self.target_column}
        feature_cols = [c for c in merged_train.columns if c not in drop_cols]

        bad_cols = [c for c in feature_cols if merged_train[c].dtype in (pl.Utf8, pl.Categorical)]
        if bad_cols:
            raise ValueError(
                f"Non-numeric columns found after pipeline processing: {bad_cols}. "
                f"Encoding step must handle all categorical columns."
            )

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

        merged_val = merged_val.select([self.id_column, self.target_column] + feature_cols)

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
