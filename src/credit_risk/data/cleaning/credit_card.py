"""Credit_card_balance table cleaner."""

from __future__ import annotations

from polars import DataFrame

from typing import override

from credit_risk.data.base import StatelessStep


class CreditCardCleaner(StatelessStep):
    """Cleaner for credit_card_balance table.

    No specific cleaning needed.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X
