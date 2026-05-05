"""Previous_application table cleaner."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class PreviousApplicationCleaner(StatelessStep):
    """Cleaner for previous_application table.

    Handles:
    - DAYS_* columns: 365243 sentinel → null
    - AMT_CREDIT_PERC: values outside (0, 10] are data errors → null
      (ratio of application credit to goods price — physically bounded)
    - String columns with XNA / XAP sentinels → null
      (XNA = missing/unknown, XAP = not applicable)
    """

    _SENTINEL = 365243
    _SENTINEL_COLS = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    _XNA_COLS = [
        "NAME_CONTRACT_TYPE",
        "NAME_CLIENT_TYPE",
        "NAME_PAYMENT_TYPE",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP",
        "CODE_REJECT_REASON",
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

        for col in self._XNA_COLS:
            if col in X.columns:
                exprs.append(
                    pl.when(pl.col(col).is_in(["XNA", "XAP"]))
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        if exprs:
            X = X.with_columns(exprs)

        return X
