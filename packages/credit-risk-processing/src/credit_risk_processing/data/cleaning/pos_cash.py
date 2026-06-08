"""POS_CASH_balance table cleaner."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk_processing.data.base import StatelessStep


class POSCashBalanceCleaner(StatelessStep):
    """Cleaner for POS_CASH_balance table.

    Cleans XNA values in contract status.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        if "NAME_CONTRACT_STATUS" in X.columns:
            X = X.with_columns(
                pl.when(pl.col("NAME_CONTRACT_STATUS") == "XNA")
                .then(pl.lit(None))
                .otherwise(pl.col("NAME_CONTRACT_STATUS"))
                .alias("NAME_CONTRACT_STATUS")
            )

        return X
