"""Data loading, cleaning, encoding, and preprocessing utilities."""

from __future__ import annotations

# Step classes and registries — from shared package
from credit_risk_processing.data import (
    AggregatorRegistry,
    CategoricalEncoder,
    CleaningRegistry,
    EncodingRegistry,
    ImputationRegistry,
    NoOpStep,
    PolarsOneHotEncoder,
    ProcessingStep,
    StatelessStep,
    TransformerRegistry,
)

# Loader — training-specific, stays local
from credit_risk.data.loader import (
    TABLE_NAMES,
    BaseDataLoader,
    PDDataLoader,
    PLDataLoader,
    PLLazyDataLoader,
    get_table_csv_names,
)

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
]
