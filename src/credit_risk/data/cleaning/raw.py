"""Raw (no-op) cleaner that passes data through unchanged."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.data.cleaning.base import TableCleaner


class RawCleaner(TableCleaner):
    """No-op cleaner that returns data unchanged."""

    def clean(self, df: DataFrame) -> DataFrame:
        return df
