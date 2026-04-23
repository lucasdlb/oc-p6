"""Bureau_balance table cleaner."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class BureauBalanceCleaner(TableCleaner):
    """Cleaner for bureau_balance table.

    No specific cleaning needed - mostly categorical STATUS and MONTHS_BALANCE.
    """

    def clean(self, df: DataFrame) -> DataFrame:
        return df
