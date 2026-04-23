"""Base protocol and reusable imputation classes for table-specific imputers."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from polars import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)

NUMERIC_TYPES = (pl.Float64, pl.Int64, pl.Float32, pl.Int32, pl.UInt32, pl.UInt64)


@runtime_checkable
class TableImputer(Protocol):
    """Protocol for table-specific imputation operations.

    Imputation should handle:
    - Domain-aware imputation (e.g., OWN_CAR_AGE = -1 when no car)
    - Median/mean imputation for numeric columns
    - Mode imputation for categorical columns
    - Missing value flags
    """

    def impute(self, df: DataFrame) -> DataFrame:
        """Impute missing values in the dataframe.

        Args:
            df: Input dataframe.

        Returns:
            Imputed dataframe.
        """
        ...


class DefaultNumericImputer:
    """Imputes numeric columns with median."""

    def impute(self, df: DataFrame) -> DataFrame:
        numeric_cols = [c for c in df.columns if df.schema[c] in NUMERIC_TYPES]
        for col in numeric_cols:
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count > 0:
                median_val = df.select(pl.col(col).median()).item() or 0.0
                df = df.with_columns(pl.col(col).fill_null(median_val).alias(col))
        return df


class CategoricalImputer:
    """Imputes string columns with mode."""

    def impute(self, df: DataFrame) -> DataFrame:
        str_cols = [c for c in df.columns if df.schema[c] == pl.String]
        for col in str_cols:
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count > 0:
                mode_val = df.select(pl.col(col).drop_nulls().mode().first()).item()
                if mode_val is not None:
                    df = df.with_columns(pl.col(col).fill_null(mode_val).alias(col))
        return df


class MedianAndModeImputer(DefaultNumericImputer, CategoricalImputer):
    """Combined imputer: numeric with median, categorical with mode."""

    def impute(self, df: DataFrame) -> DataFrame:
        df = DefaultNumericImputer.impute(self, df)
        df = CategoricalImputer.impute(self, df)
        return df


class LGBMIterativeImputer:
    """Uses LGBM + IterativeImputer for complex numeric imputation.

    More accurate than simple median but slower.
    Uses LGBMRegressor as estimator for better performance.
    """

    def __init__(self, max_iter: int = 3, n_estimators: int = 50):
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def impute(self, df: DataFrame) -> DataFrame:
        numeric_cols = [c for c in df.columns if df.schema[c] in NUMERIC_TYPES]
        if not numeric_cols:
            return df

        # Convert to pandas for sklearn
        pdf = df.select(numeric_cols).to_pandas()

        # First: simple fill for columns with low missing (<=5%)
        low_missing = [c for c in numeric_cols if pdf[c].isna().sum() / len(pdf) <= 0.05]
        if low_missing:
            pdf[low_missing] = pdf[low_missing].fillna(pdf[low_missing].median())

        # Then: iterative imputation for high-missing columns
        high_missing = [c for c in numeric_cols if c not in low_missing]
        if high_missing:
            imp = IterativeImputer(
                estimator=LGBMRegressor(
                    n_estimators=self.n_estimators, verbosity=-1, random_state=42
                ),
                max_iter=self.max_iter,
                random_state=42,
            )
            pdf[high_missing] = imp.fit_transform(pdf[high_missing])

        # Replace columns in original df
        for col in numeric_cols:
            df = df.with_columns(pl.col(col).fill_null(pdf[col].iloc[0]).alias(col))
            # Actually we need to replace all values
        df = df.with_columns(
            [
                pl.col(c).map_elements(lambda x: x, return_dtype=pl.Float64).alias(c)
                for c in numeric_cols
                if c in pdf.columns
            ]
        )

        # More efficient: get all imputed values at once
        result_df = df.clone()
        for col in numeric_cols:
            if col in pdf.columns:
                series = pl.Series(name=col, values=pdf[col].values)
                result_df = result_df.with_columns(series)

        return result_df
