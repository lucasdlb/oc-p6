"""Data transformer composite that delegates to table-specific transformers."""

from __future__ import annotations

import logging

import polars as pl
from polars import DataFrame

from credit_risk.config import FeaturesConfig
from credit_risk.data.encoder import CategoricalEncoder
from credit_risk.data.encoding import PolarsOneHotEncoder
from credit_risk.features.transformers.registry import TransformerRegistry

logger = logging.getLogger(__name__)


class DataTransformer:
    """Composite transformer that delegates to table-specific implementations.

    Applies feature engineering transformations to aggregated features,
    such as interactions, ratios, and derived features.

    Also handles encoding of categorical columns.

    Usage:
        transformer = DataTransformer()
        df = transformer.transform(df, table="bureau")
    """

    def __init__(
        self,
        config: FeaturesConfig | None = None,
    ):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    def transform(
        self,
        df: DataFrame,
        table: str = "application",
        method: str | None = None,
        encoding: str | None = None,
    ) -> DataFrame:
        """Transform features for a specific table.

        Args:
            df: Input dataframe
            table: Table name (e.g., "bureau", "bureau_balance")
            method: Transform method (default from config)
            encoding: Encoding method ("onehot", "label", "none")

        Returns:
            Transformed dataframe with engineered features
        """
        method = method or "default"
        logger.info(f"Transforming table '{table}' with method '{method}'")
        logger.info(f"  Before: {df.height} rows, {df.width} cols")

        transformer = TransformerRegistry.get_transformer(table, method)
        result = transformer.transform(df)
        logger.info(f"  After transform: {result.height} rows, {result.width} cols")

        if encoding and encoding != "none":
            result = self._encode(result, encoding)

        return result

    def _encode(self, df: DataFrame, method: str) -> DataFrame:
        """Apply encoding to categorical columns."""
        string_cols = [c for c in df.columns if df.schema[c] == pl.String]
        if not string_cols:
            logger.info("  No categorical columns to encode")
            return df

        logger.info(f"  Encoding {len(string_cols)} categorical columns with {method}")

        if method == "onehot":
            encoder = PolarsOneHotEncoder(max_categories=20)
            result = encoder.fit_transform(df)
        elif method == "label":
            encoder = CategoricalEncoder()
            result = encoder.fit_transform(df)
        else:
            result = df

        logger.info(f"  After encoding: {result.height} rows, {result.width} cols")
        return result
