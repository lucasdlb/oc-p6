"""Registry for table-specific aggregators."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from credit_risk.features.aggregators.base import TableAggregator


class AggregatorRegistry:
    """Registry for looking up table-specific aggregators."""

    _registry: dict[str, type[TableAggregator]] = {}

    @classmethod
    def get_aggregator(cls, table: str, method: str = "default") -> TableAggregator:
        """Get aggregator instance for a table.

        Args:
            table: Table name (e.g., "bureau_balance")
            method: Aggregation method ("default", "minimal", "detailed")

        Returns:
            TableAggregator instance
        """
        cls._ensure_initialized()
        if table not in cls._registry:
            registered = list(cls._registry.keys())
            raise KeyError(
                f"No aggregator registered for table '{table}'. Did you mean one of: {registered}?"
            )
        return cls._registry[table]()

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._registry:
            cls._register_defaults()

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default aggregators for all tables."""
        from credit_risk.features.aggregators.bureau import BureauAggregator
        from credit_risk.features.aggregators.bureau_balance import BureauBalanceAggregator
        from credit_risk.features.aggregators.credit_card import CreditCardAggregator
        from credit_risk.features.aggregators.installments import InstallmentsAggregator
        from credit_risk.features.aggregators.noop import NoOpAggregator
        from credit_risk.features.aggregators.pos_cash import POSCashAggregator
        from credit_risk.features.aggregators.previous_application import (
            PreviousApplicationAggregator,
        )

        cls._registry["application"] = NoOpAggregator
        cls._registry["bureau"] = BureauAggregator
        cls._registry["bureau_balance"] = BureauBalanceAggregator
        cls._registry["previous_application"] = PreviousApplicationAggregator
        cls._registry["pos_cash_balance"] = POSCashAggregator
        cls._registry["installments_payments"] = InstallmentsAggregator
        cls._registry["credit_card_balance"] = CreditCardAggregator

    @classmethod
    def register(cls, table: str, aggregator: type[TableAggregator]) -> None:
        """Register a custom aggregator for a table.

        Args:
            table: Table name (e.g., "bureau_balance")
            aggregator: Aggregator class to register
        """
        cls._ensure_initialized()
        cls._registry[table] = aggregator

    @classmethod
    @contextmanager
    def override(cls, table: str, aggregator: type[TableAggregator]) -> Generator[None, None, None]:
        """Temporarily override an aggregator for a table.

        Useful for experiments and testing.

        Args:
            table: Table name to override
            aggregator: Aggregator class to use temporarily

        Yields:
            None
        """
        cls._ensure_initialized()
        previous = cls._registry.get(table)
        cls._registry[table] = aggregator
        try:
            yield
        finally:
            if previous is None:
                cls._registry.pop(table, None)
            else:
                cls._registry[table] = previous

    @classmethod
    def available_tables(cls) -> list[str]:
        """Return list of registered table names.

        Returns:
            List of table names with registered aggregators
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())
