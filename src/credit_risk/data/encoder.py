"""Categorical encoding utilities."""

import polars as pl
from polars import DataFrame

from credit_risk.config import DataConfig


class CategoricalEncoder:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self._label_mappings: dict[str, dict[str, int]] = {}

    def get_categorical_columns(self, df: DataFrame) -> list[str]:
        return df.select(pl.col(pl.String)).columns

    def get_cardinality(self, df: DataFrame, col: str) -> int:
        return df.select(pl.col(col).n_unique()).item()

    def get_low_cardinality_columns(self, df: DataFrame) -> list[str]:
        threshold = self.config.categorical_threshold
        return [
            col
            for col in self.get_categorical_columns(df)
            if self.get_cardinality(df, col) <= threshold
        ]

    def get_high_cardinality_columns(self, df: DataFrame) -> list[str]:
        threshold = self.config.categorical_threshold
        return [
            col
            for col in self.get_categorical_columns(df)
            if self.get_cardinality(df, col) > threshold
        ]

    def label_encode(self, df: DataFrame, columns: list[str] | None = None) -> DataFrame:
        if columns is None:
            columns = self.get_low_cardinality_columns(df)

        result = df.clone()
        for col in columns:
            unique_values = df.select(pl.col(col).unique().sort()).to_series()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self._label_mappings[col] = mapping
            result = result.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int16).alias(col)
            )
        return result

    def get_feature_names(self, df: DataFrame, target_col: str, id_col: str) -> list[str]:
        return [col for col in df.columns if col not in [target_col, id_col]]

    def prepare_for_sklearn(self, df: DataFrame, target_col: str, id_col: str) -> tuple:
        feature_cols = self.get_feature_names(df, target_col, id_col)
        X = df.select(feature_cols).to_numpy()
        y = df.select(target_col).to_numpy().ravel()
        return X, y
