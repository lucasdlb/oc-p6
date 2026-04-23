"""Registry for table-specific cleaners."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from credit_risk.data.cleaning.base import TableCleaner
from credit_risk.data.cleaning.raw import RawCleaner


class CleaningRegistry:
    """Registry for looking up table-specific cleaners."""

    _registry: dict[str, type[TableCleaner]] = {}

    @classmethod
    def get_cleaner(cls, table: str, method: str = "default") -> TableCleaner:
        """Get cleaner instance for a table.

        Args:
            table: Table name (e.g., "bureau", "application")
            method: Cleaning method ("default" or "raw")

        Returns:
            TableCleaner instance
        """
        cls._ensure_initialized()
        if method == "raw":
            return RawCleaner()
        if table not in cls._registry:
            registered = list(cls._registry.keys())
            raise KeyError(
                f"No cleaner registered for table '{table}'. Did you mean one of: {registered}?"
            )
        return cls._registry[table]()

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._registry:
            cls._register_defaults()

    @classmethod
    def available_tables(cls) -> list[str]:
        """Return list of registered table names.

        Returns:
            List of table names with registered cleaners
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default cleaners for all tables."""
        from credit_risk.data.cleaning.application import ApplicationCleaner
        from credit_risk.data.cleaning.bureau import BureauCleaner
        from credit_risk.data.cleaning.bureau_balance import BureauBalanceCleaner
        from credit_risk.data.cleaning.credit_card import CreditCardCleaner
        from credit_risk.data.cleaning.installments import InstallmentsCleaner
        from credit_risk.data.cleaning.pos_cash import POSCashCleaner
        from credit_risk.data.cleaning.previous_application import PreviousApplicationCleaner

        cls._registry["application"] = ApplicationCleaner
        cls._registry["bureau"] = BureauCleaner
        cls._registry["bureau_balance"] = BureauBalanceCleaner
        cls._registry["previous_application"] = PreviousApplicationCleaner
        cls._registry["pos_cash_balance"] = POSCashCleaner
        cls._registry["installments_payments"] = InstallmentsCleaner
        cls._registry["credit_card_balance"] = CreditCardCleaner

    @classmethod
    def register(cls, table: str, cleaner: type[TableCleaner]) -> None:
        """Register a custom cleaner for a table.

        Args:
            table: Table name (e.g., "bureau", "application")
            cleaner: Cleaner class to register
        """
        cls._ensure_initialized()
        cls._registry[table] = cleaner

    @classmethod
    @contextmanager
    def override(cls, table: str, cleaner: type[TableCleaner]) -> Generator[None, None, None]:
        """Temporarily override a cleaner for a table.

        Useful for experiments and testing.

        Args:
            table: Table name to override
            cleaner: Cleaner class to use temporarily

        Yields:
            None
        """
        cls._ensure_initialized()
        previous = cls._registry.get(table)
        cls._registry[table] = cleaner
        try:
            yield
        finally:
            if previous is None:
                cls._registry.pop(table, None)
            else:
                cls._registry[table] = previous
