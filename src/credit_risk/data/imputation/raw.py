"""Raw (no-op) imputer that passes data through unchanged."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.data.imputation.base import TableImputer


class RawImputer(TableImputer):
    """No-op imputer that returns data unchanged."""

    def impute(self, df: DataFrame) -> DataFrame:
        return df
