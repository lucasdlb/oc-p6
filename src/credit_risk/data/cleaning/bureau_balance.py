"""Bureau_balance table cleaner."""

from __future__ import annotations

from typing import override

from polars import DataFrame

from credit_risk.data.base import StatelessStep


class BureauBalanceCleaner(StatelessStep):
    """Cleaner for bureau_balance table.

    No specific cleaning needed - mostly categorical STATUS and MONTHS_BALANCE.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X
