"""Table-specific aggregators."""

from credit_risk.features.aggregators.base import TableAggregator
from credit_risk.features.aggregators.registry import AggregatorRegistry

__all__ = ["TableAggregator", "AggregatorRegistry"]
