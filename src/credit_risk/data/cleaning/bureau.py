"""Bureau table cleaner."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class BureauCleaner(StatelessStep):
    """Cleaner for bureau table.

    DAYS_* columns use days relative to application date:
      - Negative  = in the past (normal)
      - Positive  = in the future / still active (normal)
      - +365243   = sentinel "unknown/never" (same artifact as DAYS_EMPLOYED)
      - -1        = NOT a sentinel here — it means "ended 1 day before application"

    AMT columns: negatives and extreme outliers are data errors.
    """

    _SENTINEL = 365243
    _SENTINEL_COLS = [
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
    ]
    _AMT_COLS = [
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
    ]

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        exprs = []

        for col in self._SENTINEL_COLS:
            if col in X.columns:
                exprs.append(
                    pl.when(pl.col(col).abs() == self._SENTINEL)
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        for col in self._AMT_COLS:
            if col in X.columns:
                if X.schema[col] == pl.String:
                    X = X.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
                exprs.append(
                    pl.when(pl.col(col) < 0).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
                )

        if "CREDIT_DAY_OVERDUE" in X.columns:
            if X.schema["CREDIT_DAY_OVERDUE"] == pl.String:
                X = X.with_columns(
                    pl.col("CREDIT_DAY_OVERDUE")
                    .cast(pl.Float64, strict=False)
                    .alias("CREDIT_DAY_OVERDUE")
                )
            exprs.append(
                pl.when(pl.col("CREDIT_DAY_OVERDUE") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("CREDIT_DAY_OVERDUE"))
                .alias("CREDIT_DAY_OVERDUE")
            )

        if "CNT_CREDIT_PROLONG" in X.columns:
            if X.schema["CNT_CREDIT_PROLONG"] == pl.String:
                X = X.with_columns(
                    pl.col("CNT_CREDIT_PROLONG")
                    .cast(pl.Float64, strict=False)
                    .alias("CNT_CREDIT_PROLONG")
                )
            exprs.append(
                pl.when(pl.col("CNT_CREDIT_PROLONG") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("CNT_CREDIT_PROLONG"))
                .alias("CNT_CREDIT_PROLONG")
            )

        if exprs:
            X = X.with_columns(exprs)

        return X
