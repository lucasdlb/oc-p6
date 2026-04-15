"""Data transformer composite that delegates to table-specific transformers."""

from __future__ import annotations

import logging

from polars import DataFrame

from credit_risk.config import DataSourcesConfig, FeaturesConfig
from credit_risk.features.transformers.registry import TransformerRegistry

logger = logging.getLogger(__name__)


class DataTransformer:
    """Composite transformer that delegates to table-specific implementations.

    Applies feature engineering transformations to aggregated features,
    such as interactions, ratios, and derived features.

    Usage:
        transformer = DataTransformer(data_sources=ds_config)
        df = transformer.transform(df, table="bureau")
    """

    def __init__(
        self,
        config: FeaturesConfig | None = None,
        data_sources: DataSourcesConfig | None = None,
    ):
        from credit_risk.config import cfg

        self.config = config or cfg.data.features
        self.data_sources = data_sources

    def transform(self, df: DataFrame, table: str = "application") -> DataFrame:
        """Transform features for a specific table.

        Args:
            df: Input dataframe
            table: Table name (e.g., "bureau", "bureau_balance")

        Returns:
            Transformed dataframe with engineered features
        """
        method = self._get_transform_method(table)
        logger.info(f"Transforming table '{table}' with method '{method}'")
        logger.info(f"  Before: {df.height} rows, {df.width} cols")
        transformer = TransformerRegistry.get_transformer(table, method)
        result = transformer.transform(df)
        logger.info(f"  After: {result.height} rows, {result.width} cols")
        return result

    def _get_transform_method(self, table: str) -> str:
        """Get transform method for a table from data_sources config."""
        if self.data_sources is None:
            return "default"
        return self.data_sources.get_transform_method(table)
