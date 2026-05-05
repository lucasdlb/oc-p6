"""Reusable imputation strategies for table-specific imputers."""

from __future__ import annotations

import logging
from typing import Self

import polars as pl
from lightgbm import LGBMRegressor
from polars import DataFrame
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from credit_risk.data.base import ProcessingStep

logger = logging.getLogger(__name__)

NUMERIC_TYPES = (pl.Float64, pl.Int64, pl.Float32, pl.Int32, pl.UInt32, pl.UInt64)


class DefaultNumericImputer(ProcessingStep):
    """Imputes numeric columns with median."""

    def __init__(self) -> None:
        self._medians: dict[str, float] = {}

    def fit(self, X: DataFrame, y=None) -> Self:
        self._medians = {}
        for col in X.columns:
            if X.schema[col] in NUMERIC_TYPES:
                self._medians[col] = X.select(pl.col(col).median()).item() or 0.0
        return self

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        exprs = []
        for col, median in self._medians.items():
            if col in X.columns and X.schema[col] in NUMERIC_TYPES:
                original_dtype = X.schema[col]
                filled = pl.col(col).fill_null(median)
                # Preserve integer types
                if original_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    filled = filled.cast(pl.Int64)
                exprs.append(filled.alias(col))
        if exprs:
            X = X.with_columns(exprs)
        return X


class CategoricalImputer(ProcessingStep):
    """Imputes string columns with mode."""

    def __init__(self) -> None:
        self._modes: dict[str, str] = {}

    def fit(self, X: DataFrame, y=None) -> Self:
        self._modes = {}
        for col in X.columns:
            if X.schema[col] == pl.String:
                mode_val = X.select(pl.col(col).drop_nulls().mode().first()).item()
                if mode_val is not None:
                    self._modes[col] = mode_val
        return self

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        exprs = []
        for col, mode in self._modes.items():
            if col in X.columns:
                exprs.append(pl.col(col).fill_null(mode).alias(col))
        if exprs:
            X = X.with_columns(exprs)
        return X


class MedianAndModeImputer(ProcessingStep):
    """Combined imputer: numeric with median, categorical with mode."""

    def __init__(self) -> None:
        self._numeric_imputer = DefaultNumericImputer()
        self._categorical_imputer = CategoricalImputer()

    def fit(self, X: DataFrame, y=None) -> Self:
        self._numeric_imputer.fit(X)
        self._categorical_imputer.fit(X)
        return self

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        X = self._numeric_imputer.transform(X)
        X = self._categorical_imputer.transform(X)
        return X


class LGBMIterativeImputer(ProcessingStep):
    """Uses LGBM + IterativeImputer for complex numeric imputation.

    More accurate than simple median but slower.
    Uses LGBMRegressor as estimator for better performance.
    """

    def __init__(self, max_iter: int = 3, n_estimators: int = 50) -> None:
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self._imputer: IterativeImputer | None = None
        self._numeric_cols: list[str] = []
        self._int_cols: dict[str, object] = {}
        self._pdf: dict[str, object] | None = None

    def fit(self, X: DataFrame, y=None) -> Self:
        numeric_cols = [c for c in X.columns if X.schema[c] in NUMERIC_TYPES]
        if not numeric_cols:
            return self

        self._numeric_cols = numeric_cols
        self._int_cols = {
            col: dtype
            for col, dtype in zip(X.columns, X.dtypes, strict=True)
            if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        }

        pdf = X.select(numeric_cols).to_pandas()
        low_missing = [c for c in numeric_cols if pdf[c].isna().sum() / len(pdf) <= 0.05]
        high_missing = [c for c in numeric_cols if c not in low_missing]

        if low_missing:
            pdf[low_missing] = pdf[low_missing].fillna(pdf[low_missing].median())

        if high_missing:
            imp = IterativeImputer(
                estimator=LGBMRegressor(
                    n_estimators=self.n_estimators, verbosity=-1, random_state=42
                ),
                max_iter=self.max_iter,
                random_state=42,
            )
            pdf[high_missing] = imp.fit_transform(pdf[high_missing])
            self._imputer = imp

        self._pdf = {"data": pdf}
        return self

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        if not self._numeric_cols:
            return X

        pdf = X.select(self._numeric_cols).to_pandas()
        result_X = X.clone()

        if self._pdf is not None:
            low_missing = [c for c in self._numeric_cols if pdf[c].isna().sum() / len(pdf) <= 0.05]
            high_missing = [c for c in self._numeric_cols if c not in low_missing]
            if low_missing:
                train_pdf = self._pdf["data"]
                pdf[low_missing] = pdf[low_missing].fillna(train_pdf[low_missing].median())
            if high_missing and self._imputer is not None:
                pdf[high_missing] = self._imputer.transform(pdf[high_missing])

        for col in self._numeric_cols:
            if col in pdf.columns:
                series = pl.Series(name=col, values=pdf[col].values)
                result_X = result_X.with_columns(series)

        return result_X
