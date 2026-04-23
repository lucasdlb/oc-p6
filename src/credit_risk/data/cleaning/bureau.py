"""Bureau table cleaner."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class BureauCleaner(TableCleaner):
    """Cleaner for bureau table.

    DAYS_* columns use days relative to application date:
      - Negative  = in the past (normal)
      - Positive  = in the future / still active (normal)
      - +365243   = sentinel "unknown/never" (same artifact as DAYS_EMPLOYED)
      - -1        = NOT a sentinel here — it means "ended 1 day before application"

    AMT columns: negatives and extreme outliers are data errors.
    """

    # Sentinel used across multiple DAYS columns in this dataset
    _SENTINEL = 365243

    # Columns where the sentinel appears
    _SENTINEL_COLS = [
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
    ]

    # Plausible bounds for credit amounts — beyond these are data errors
    _AMT_COLS = [
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
    ]

    def clean(self, df: DataFrame) -> DataFrame:
        exprs = []

        # --- sentinel handling ---
        for col in self._SENTINEL_COLS:
            if col in df.columns:
                exprs.append(
                    pl.when(pl.col(col).abs() == self._SENTINEL)
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        # --- negative AMT values are data errors → null ---
        for col in self._AMT_COLS:
            if col in df.columns:
                exprs.append(
                    pl.when(pl.col(col) < 0).then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
                )

        # --- CREDIT_DAY_OVERDUE: negative makes no sense → null ---
        if "CREDIT_DAY_OVERDUE" in df.columns:
            exprs.append(
                pl.when(pl.col("CREDIT_DAY_OVERDUE") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("CREDIT_DAY_OVERDUE"))
                .alias("CREDIT_DAY_OVERDUE")
            )

        # --- CNT_CREDIT_PROLONG: shouldn't be negative ---
        if "CNT_CREDIT_PROLONG" in df.columns:
            exprs.append(
                pl.when(pl.col("CNT_CREDIT_PROLONG") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("CNT_CREDIT_PROLONG"))
                .alias("CNT_CREDIT_PROLONG")
            )

        if exprs:
            df = df.with_columns(exprs)

        return df
