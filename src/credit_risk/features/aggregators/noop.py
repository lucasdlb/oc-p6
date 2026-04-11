"""No-op aggregator for tables that don't need aggregation."""

from __future__ import annotations

from polars import LazyFrame

from credit_risk.features.aggregators.base import TableAggregator


class NoOpAggregator(TableAggregator):
    """No-op aggregator for tables that are already at SK_ID_CURR level.

    Used for application_train which doesn't need aggregation.
    """

    @classmethod
    def load_link(cls) -> LazyFrame | None:
        return None

    def aggregate(self, df: LazyFrame, method: str = "default") -> LazyFrame:
        """Return df as-is (no aggregation needed)."""
        return df
