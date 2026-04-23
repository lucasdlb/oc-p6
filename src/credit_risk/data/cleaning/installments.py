"""Installments_payments table cleaner."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class InstallmentsCleaner(TableCleaner):
    """Cleaner for installments_payments table.

    No specific cleaning needed.
    """

    def clean(self, df: DataFrame) -> DataFrame:

        return df
