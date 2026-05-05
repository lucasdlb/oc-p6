"""Installments_payments table cleaner."""

from __future__ import annotations

from typing import override

from polars import DataFrame

from credit_risk.data.base import StatelessStep


class InstallmentsCleaner(StatelessStep):
    """Cleaner for installments_payments table.

    No specific cleaning needed.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X
