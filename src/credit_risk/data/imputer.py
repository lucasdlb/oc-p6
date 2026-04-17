"""Data imputer composite that delegates to table-specific imputers."""

from __future__ import annotations

import logging

from polars import DataFrame

from credit_risk.config import DataConfig
from credit_risk.data.imputation.registry import ImputationRegistry

logger = logging.getLogger(__name__)


class DataImputer:
    """Composite imputer that delegates to table-specific implementations.

    Usage:
        imputer = DataImputer(data_sources=ds_config)
        df = imputer.impute(df, table="application")
    """

    def __init__(
        self,
        config: DataConfig | None = None,
        data_sources: DataSourcesConfig | None = None,
    ):
        from credit_risk.config import load_config

        self.config = config or load_config().data
        self.data_sources = data_sources

    def impute(
        self, df: DataFrame, table: str = "application", method: str | None = None
    ) -> DataFrame:
        """Impute missing values for a specific table.

        Args:
            df: Input dataframe.
            table: Table name (e.g., "application", "bureau")
            method: Imputation method (default from config)

        Returns:
            Imputed dataframe.
        """
        method = method or self._get_imputation_method(table)
        nulls_before = df.null_count().to_numpy().sum()
        logger.info(f"Imputing table '{table}' with method '{method}'")
        logger.info(f"  Before: {df.height} rows, {df.width} cols, {nulls_before} nulls")
        imputer = ImputationRegistry.get_imputer(table, method)
        result = imputer.impute(df)
        nulls_after = result.null_count().to_numpy().sum()
        logger.info(f"  After: {result.height} rows, {result.width} cols, {nulls_after} nulls")
        return result

    def _get_imputation_method(self, table: str) -> str:
        """Get imputation method for a table from data_sources config."""
        if self.data_sources is None:
            return "default"
        return self.data_sources.get_imputation_method(table)
