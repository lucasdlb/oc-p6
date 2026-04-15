"""Experiment configuration using ConfigGrid.

This module provides the DataSourcesConfig class for experiment grids.
Uses ConfigGrid for hyperparameter sweeps.
"""

from __future__ import annotations

from pydantic import BaseModel


class DataSourcesConfig(BaseModel):
    """Configuration for which data sources to use."""

    application: bool = True
    bureau: bool = True
    bureau_balance: bool = True
    previous_application: bool = True
    pos_cash_balance: bool = True
    installments_payments: bool = True
    credit_card_balance: bool = True

    def is_enabled(self, source: str) -> bool:
        """Check if a data source is enabled."""
        return getattr(self, source, False)

    def get_enabled_sources(self) -> list[str]:
        """Get list of enabled source names."""
        return [k for k, v in self.model_dump().items() if v is True]
