"""Backward compatibility - DataSourcesConfig for experiment grid.

This module provides the DataSourcesConfig class that was removed from
the old experiment_config.py. New code should use the TOML-based config system.
"""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field


class DataSourceSpec(BaseModel):
    """Specification for a data source."""

    enabled: bool = True
    cleaning_method: str = "default"
    imputation_method: str = "default"
    transform_method: str = "default"


DataSourceValue = Union[bool, list[bool], DataSourceSpec]


class DataSourcesConfig(BaseModel):
    """Configuration for which data sources to use."""

    application: DataSourceValue = True
    bureau: DataSourceValue = True
    bureau_balance: DataSourceValue = True
    previous_application: DataSourceValue = True
    pos_cash_balance: DataSourceValue = True
    installments_payments: DataSourceValue = True
    credit_card_balance: DataSourceValue = True

    def is_enabled(self, source: str) -> bool:
        """Check if a data source is enabled."""
        value = getattr(self, source, None)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, list):
            return value[0] if value else False
        return value.enabled

    def is_grid(self, source: str) -> bool:
        """Check if a data source has multiple values (grid search)."""
        value = getattr(self, source, None)
        return isinstance(value, list) and len(value) > 1

    def get_grid_values(self, source: str) -> list[bool]:
        """Get grid values for a data source."""
        value = getattr(self, source, None)
        if isinstance(value, list):
            return value
        if isinstance(value, bool):
            return [value]
        if isinstance(value, DataSourceSpec):
            return [value.enabled]
        return [True]

    def get_cleaning_method(self, source: str) -> str:
        """Get cleaning method for a data source."""
        value = getattr(self, source, None)
        if value is None:
            return "default"
        if isinstance(value, bool):
            return "default"
        if isinstance(value, list):
            return "default"
        return value.cleaning_method

    def get_imputation_method(self, source: str) -> str:
        """Get imputation method for a data source."""
        value = getattr(self, source, None)
        if value is None:
            return "default"
        if isinstance(value, bool):
            return "default"
        if isinstance(value, list):
            return "default"
        return value.imputation_method

    def get_transform_method(self, source: str) -> str:
        """Get transform method for a data source."""
        value = getattr(self, source, None)
        if value is None:
            return "default"
        if isinstance(value, bool):
            return "default"
        if isinstance(value, list):
            return "default"
        return value.transform_method

    def get_single_values(self) -> dict[str, bool]:
        """Get a dict of single values (non-grid sources)."""
        result = {}
        for field in self.model_fields:
            value = getattr(self, field)
            if isinstance(value, bool):
                result[field] = value
            elif isinstance(value, list) and len(value) == 1:
                result[field] = value[0]
            elif isinstance(value, DataSourceSpec):
                result[field] = value.enabled
        return result


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    name: str = "experiment"
    description: str = ""
    model: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    aggregation: dict[str, str] = Field(default_factory=dict)
    search: dict[str, list[Any]] = Field(default_factory=dict)
