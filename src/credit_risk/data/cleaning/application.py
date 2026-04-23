"""Application table cleaner."""

from __future__ import annotations

import logging

import polars as pl
from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner

logger = logging.getLogger(__name__)


class ApplicationCleaner(TableCleaner):
    """Cleaner for application_train/test tables.

    Handles:
    - DAYS_EMPLOYED = 365243  → flag + null + derive YEARS_EMPLOYED, drop raw
    - CODE_GENDER = "XNA"     → null (3 rows, data entry error)
    """

    _SENTINEL = 365243

    def clean(self, df: DataFrame) -> DataFrame:
        original_cols = set(df.columns)
        exprs = []
        drop_cols = []

        if "DAYS_EMPLOYED" in df.columns:
            is_sentinel = pl.col("DAYS_EMPLOYED") == self._SENTINEL
            is_anomalous = is_sentinel | (pl.col("DAYS_EMPLOYED") > 0)
            exprs += [
                is_sentinel.cast(pl.Int8).alias("is_never_employed"),
                pl.when(is_anomalous)
                .then(pl.lit(None))
                .otherwise(pl.col("DAYS_EMPLOYED") / -365.25)
                .alias("YEARS_EMPLOYED"),
            ]
            drop_cols.append("DAYS_EMPLOYED")

        if "CODE_GENDER" in df.columns:
            exprs.append(
                pl.when(pl.col("CODE_GENDER") == "XNA")
                .then(pl.lit(None))
                .otherwise(pl.col("CODE_GENDER"))
                .alias("CODE_GENDER")
            )

        if exprs:
            df = df.with_columns(exprs)
        if drop_cols:
            df = df.drop(drop_cols)

        return df
