"""Base protocol for table-specific cleaners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from polars import DataFrame


@runtime_checkable
class TableCleaner(Protocol):
    """Protocol for table-specific cleaning operations.

    Cleaning should handle:
    - Invalid values (e.g., 365243 days -> null)
    - Data type conversions
    - Domain-specific cleaning rules
    """

    def clean(self, df: DataFrame) -> DataFrame:
        """Clean the dataframe.

        Args:
            df: Input dataframe.

        Returns:
            Cleaned dataframe.
        """
        ...
