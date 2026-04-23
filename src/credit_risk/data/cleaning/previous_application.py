"""Previous_application table cleaner."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class PreviousApplicationCleaner(TableCleaner):
    """Cleaner for previous_application table.

    Handles:
    - DAYS_* columns: 365243 sentinel → null
    - AMT_CREDIT_PERC: values outside (0, 10] are data errors → null
      (ratio of application credit to goods price — physically bounded)
    """

    _SENTINEL = 365243

    _SENTINEL_COLS = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]

    def clean(self, df: DataFrame) -> DataFrame:
        exprs = []

        # --- sentinel handling — all in one pass ---
        for col in self._SENTINEL_COLS:
            if col in df.columns:
                exprs.append(
                    pl.when(pl.col(col) == self._SENTINEL)
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        # --- APP_CREDIT_PERC: ratio credit/goods, physically bounded ---
        if "APP_CREDIT_PERC" in df.columns:
            exprs.append(
                pl.when(
                    (pl.col("APP_CREDIT_PERC") <= 0) | (pl.col("APP_CREDIT_PERC") > 10)
                )
                .then(pl.lit(None))
                .otherwise(pl.col("APP_CREDIT_PERC"))
                .alias("APP_CREDIT_PERC")
            )

        if exprs:
            df = df.with_columns(exprs)

        return df
