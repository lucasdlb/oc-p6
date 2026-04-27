"""Data loading, cleaning, encoding, and preprocessing utilities."""

from __future__ import annotations

from credit_risk.data.aggregation import AggregatorRegistry
from credit_risk.data.base import NoOpStep, ProcessingStep, StatelessStep
from credit_risk.data.cleaning import CleaningRegistry
from credit_risk.data.encoding import CategoricalEncoder, EncodingRegistry, PolarsOneHotEncoder
from credit_risk.data.imputation import ImputationRegistry
from credit_risk.data.loader import (
    TABLE_NAMES,
    BaseDataLoader,
    PDDataLoader,
    PLDataLoader,
    PLLazyDataLoader,
    get_table_csv_names,
)
from credit_risk.data.store import FeatureStore
from credit_risk.data.transformation import TransformerRegistry

__all__ = [
    # Base
    "NoOpStep",
    "ProcessingStep",
    "StatelessStep",
    # Loader
    "TABLE_NAMES",
    "BaseDataLoader",
    "PDDataLoader",
    "PLDataLoader",
    "PLLazyDataLoader",
    "get_table_csv_names",
    # Registries
    "CleaningRegistry",
    "ImputationRegistry",
    "AggregatorRegistry",
    "TransformerRegistry",
    "EncodingRegistry",
    # Encoding
    "CategoricalEncoder",
    "PolarsOneHotEncoder",
    # Store
    "FeatureStore",
]
