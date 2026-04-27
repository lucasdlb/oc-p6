"""Installments_payments table cleaner."""

from __future__ import annotations

from polars import DataFrame

from typing import override

from credit_risk.data.base import StatelessStep


class InstallmentsCleaner(StatelessStep):
    """Cleaner for installments_payments table.

    No specific cleaning needed.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X
