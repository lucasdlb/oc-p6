"""Transformer protocol for feature engineering on aggregated data."""

from __future__ import annotations

from typing import Protocol

from polars import DataFrame


class TableTransformer(Protocol):
    """Protocol for table-specific feature transformations.

    Transformations are applied to features that don't need aggregation,
    such as:
    - Feature interactions (products, ratios)
    - Binning (continuous to categorical)
    - Derived features
    - Scaling

    Usage:
        class MyTransformer:
            def transform(self, df: DataFrame) -> DataFrame:
                # Feature engineering logic
                return df
    """

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform features.

        Args:
            df: Input dataframe

        Returns:
            Transformed dataframe
        """
        ...
