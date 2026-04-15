"""Data aggregator composite that delegates to table-specific aggregators."""

from __future__ import annotations

import logging

from polars import LazyFrame

from credit_risk.config import FeaturesConfig
from credit_risk.features.aggregators.registry import AggregatorRegistry

logger = logging.getLogger(__name__)


class DataAggregator:
    """Composite aggregator that delegates to table-specific implementations.

    Usage:
        aggregator = DataAggregator(config=feature_config)
        result = aggregator.aggregate(lazy_df, table="bureau_balance")
    """

    def __init__(
        self,
        config: FeaturesConfig | None = None,
    ):
        from credit_risk.config import cfg

        self.config = config or cfg.data.features

    def aggregate(
        self, df: LazyFrame, table: str = "bureau", method: str | None = None
    ) -> LazyFrame:
        """Aggregate data for a specific table.

        Args:
            df: Input lazyframe.
            table: Table name (e.g., "bureau_balance", "bureau")
            method: Aggregation method override ("minimal", "default", "detailed")

        Returns:
            Aggregated lazyframe with SK_ID_CURR as key.
        """
        method = method or "detailed"
        logger.info(f"Aggregating table '{table}' with method '{method}'")
        agg = AggregatorRegistry.get_aggregator(table)
        result = agg.aggregate(df, method)
        collected = result.collect()
        logger.info(f"  Aggregated: {collected.height} rows, {collected.width} cols")
        return result
