"""Previous_application table cleaner."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from typing import override

from credit_risk.data.base import StatelessStep


class PreviousApplicationCleaner(StatelessStep):
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

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        exprs = []

        for col in self._SENTINEL_COLS:
            if col in X.columns:
                exprs.append(
                    pl.when(pl.col(col) == self._SENTINEL)
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        if "APP_CREDIT_PERC" in X.columns:
            exprs.append(
                pl.when((pl.col("APP_CREDIT_PERC") <= 0) | (pl.col("APP_CREDIT_PERC") > 10))
                .then(pl.lit(None))
                .otherwise(pl.col("APP_CREDIT_PERC"))
                .alias("APP_CREDIT_PERC")
            )

        if exprs:
            X = X.with_columns(exprs)

        return X
