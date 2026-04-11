"""Base protocol for table-specific aggregators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from polars import LazyFrame


@runtime_checkable
class TableAggregator(Protocol):
    """Protocol for table-specific aggregation operations.

    Aggregation should handle:
    - Computing per-ID features from auxiliary tables
    - Linking to main table via ID columns
    - Time-based aggregations (last N months, trends)
    - Method control: "default", "minimal", "detailed"
    """

    def aggregate(self, df: LazyFrame, method: str = "default") -> LazyFrame:
        """Aggregate the dataframe to SK_ID_CURR level.

        Args:
            df: Input lazyframe (table to aggregate).
            method: Aggregation method ("default", "minimal", "detailed")

        Returns:
            Aggregated lazyframe with SK_ID_CURR as key.
        """
        ...

    @classmethod
    def load_link(cls) -> LazyFrame | None:
        """Load link table for this table if needed.

        Returns:
            Link table LazyFrame, or None if table has SK_ID_CURR directly.
        """
        ...
