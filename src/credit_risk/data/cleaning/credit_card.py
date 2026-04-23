"""Credit_card_balance table cleaner."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class CreditCardCleaner(TableCleaner):
    """Cleaner for credit_card_balance table.

    No specific cleaning needed.
    """

    def clean(self, df: DataFrame) -> DataFrame:
        return df
