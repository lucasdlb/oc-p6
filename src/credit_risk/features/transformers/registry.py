"""Registry for table-specific transformers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from credit_risk.features.transformers.base import TableTransformer


class TransformerRegistry:
    """Registry for looking up table-specific transformers."""

    _registry: dict[str, type[TableTransformer]] = {}

    @classmethod
    def get_transformer(cls, table: str, method: str = "default") -> TableTransformer:
        """Get transformer instance for a table.

        Args:
            table: Table name (e.g., "bureau", "bureau_balance")
            method: Transform method ("default", "minimal", "detailed")

        Returns:
            TableTransformer instance
        """
        cls._ensure_initialized()
        if table not in cls._registry:
            registered = list(cls._registry.keys())
            raise KeyError(
                f"No transformer registered for table '{table}'. Did you mean one of: {registered}?"
            )
        return cls._registry[table]()

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._registry:
            cls._register_defaults()

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default transformers for all tables."""
        from credit_risk.features.transformers.application import ApplicationTransformer
        from credit_risk.features.transformers.bureau import BureauTransformer
        from credit_risk.features.transformers.bureau_balance import BureauBalanceTransformer
        from credit_risk.features.transformers.credit_card import CreditCardTransformer
        from credit_risk.features.transformers.installments import InstallmentsTransformer
        from credit_risk.features.transformers.pos_cash import POSCashTransformer
        from credit_risk.features.transformers.previous_application import (
            PreviousApplicationTransformer,
        )

        cls._registry["application"] = ApplicationTransformer
        cls._registry["bureau"] = BureauTransformer
        cls._registry["bureau_balance"] = BureauBalanceTransformer
        cls._registry["previous_application"] = PreviousApplicationTransformer
        cls._registry["pos_cash_balance"] = POSCashTransformer
        cls._registry["installments_payments"] = InstallmentsTransformer
        cls._registry["credit_card_balance"] = CreditCardTransformer

    @classmethod
    def register(cls, table: str, transformer: type[TableTransformer]) -> None:
        """Register a custom transformer for a table.

        Args:
            table: Table name (e.g., "bureau", "bureau_balance")
            transformer: Transformer class to register
        """
        cls._ensure_initialized()
        cls._registry[table] = transformer

    @classmethod
    @contextmanager
    def override(
        cls, table: str, transformer: type[TableTransformer]
    ) -> Generator[None, None, None]:
        """Temporarily override a transformer for a table.

        Useful for experiments and testing.

        Args:
            table: Table name to override
            transformer: Transformer class to use temporarily

        Yields:
            None
        """
        cls._ensure_initialized()
        previous = cls._registry.get(table)
        cls._registry[table] = transformer
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
            List of table names with registered transformers
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())
