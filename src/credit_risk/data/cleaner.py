"""Data cleaning utilities using Polars."""

import polars as pl
from polars import DataFrame

from credit_risk.config.settings import DataConfig


class DataCleaner:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def clean_application(self, df: DataFrame) -> DataFrame:
        if "DAYS_EMPLOYED" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("DAYS_EMPLOYED") == 365243)
                .then(pl.lit(None))
                .otherwise(pl.col("DAYS_EMPLOYED"))
                .alias("DAYS_EMPLOYED")
            )

        if "DAYS_LAST_PHONE_CHANGE" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("DAYS_LAST_PHONE_CHANGE") == 365243)
                .then(pl.lit(None))
                .otherwise(pl.col("DAYS_LAST_PHONE_CHANGE"))
                .alias("DAYS_LAST_PHONE_CHANGE")
            )

        return df

    def remove_outliers(
        self, df: DataFrame, column: str, lower: float = 0.01, upper: float = 0.99
    ) -> DataFrame:
        quantiles = df.select(pl.col(column).quantile([lower, upper]).alias("q"))
        lower_val = quantiles[0, "q"]
        upper_val = quantiles[1, "q"]
        return df.filter((pl.col(column) >= lower_val) & (pl.col(column) <= upper_val))

    def fill_missing_numeric(self, df: DataFrame, strategy: str = "median") -> DataFrame:
        numeric_cols = [
            c for c in df.columns if df.schema[c] in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
        ]
        for col in numeric_cols:
            if strategy == "median":
                median_val = df.select(pl.col(col).median()).item()
                df = df.with_columns(pl.col(col).fill_null(median_val).alias(col))
            elif strategy == "mean":
                mean_val = df.select(pl.col(col).mean()).item()
                df = df.with_columns(pl.col(col).fill_null(mean_val).alias(col))
            elif strategy == "zero":
                df = df.with_columns(pl.col(col).fill_null(0).alias(col))
        return df

    def clean(self, df: DataFrame) -> DataFrame:
        return self.clean_application(df)
