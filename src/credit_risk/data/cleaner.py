"""Data cleaner composite that delegates to table-specific cleaners."""

from __future__ import annotations

import logging

from polars import DataFrame

from credit_risk.config import DataConfig
from credit_risk.data.cleaning.registry import CleaningRegistry

logger = logging.getLogger(__name__)


class DataCleaner:
    """Composite cleaner that delegates to table-specific implementations.

    Usage:
        cleaner = DataCleaner(data_sources=ds_config)
        df = cleaner.clean(df, table="application")
    """

    def __init__(
        self,
        config: DataConfig | None = None,
        data_sources: DataSourcesConfig | None = None,
    ):
        from credit_risk.config import load_config

        self.config = config or load_config().data
        self.data_sources = data_sources

    def clean(
        self, df: DataFrame, table: str = "application", method: str | None = None
    ) -> DataFrame:
        """Clean data for a specific table.

        Args:
            df: Input dataframe.
            table: Table name (e.g., "application", "bureau")
            method: Cleaning method (default from config)

        Returns:
            Cleaned dataframe.
        """
        method = method or self._get_cleaning_method(table)
        logger.info(f"Cleaning table '{table}' with method '{method}'")
        logger.info(f"  Before: {df.height} rows, {df.width} cols")
        original_cols = set(df.columns)
        cleaner = CleaningRegistry.get_cleaner(table, method)
        result = cleaner.clean(df)

        new_cols = set(result.columns)
        added = new_cols - original_cols
        removed = original_cols - new_cols
        if added:
            logger.info(f"  Added columns: {sorted(added)}")
        if removed:
            logger.info(f"  Removed columns: {sorted(removed)}")

        logger.info(f"  After: {result.height} rows, {result.width} cols")
        return result

    def _get_cleaning_method(self, table: str) -> str:
        """Get cleaning method for a table from data_sources config."""
        if self.data_sources is None:
            return "default"
        return self.data_sources.get_cleaning_method(table)
