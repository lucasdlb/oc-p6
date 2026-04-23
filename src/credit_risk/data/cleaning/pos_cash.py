"""POS_CASH_balance table cleaner."""

from __future__ import annotations

from polars import DataFrame
import polars as pl

from credit_risk.data.cleaning.base import TableCleaner

class POSCashCleaner(TableCleaner):
    """Cleaner for POS_CASH_balance table.

    No specific cleaning needed.
    """

    def clean(self, df: DataFrame) -> DataFrame:
        if "NAME_CONTRACT_STATUS" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("NAME_CONTRACT_STATUS") == "XNA")
                .then(pl.lit(None))
                .otherwise(pl.col("NAME_CONTRACT_STATUS"))
                .alias("NAME_CONTRACT_STATUS")
            )

        return df
