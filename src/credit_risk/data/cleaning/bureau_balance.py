"""Bureau_balance table cleaner."""

from __future__ import annotations

from polars import DataFrame

from typing import override

from credit_risk.data.base import StatelessStep


class BureauBalanceCleaner(StatelessStep):
    """Cleaner for bureau_balance table.

    No specific cleaning needed - mostly categorical STATUS and MONTHS_BALANCE.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X
