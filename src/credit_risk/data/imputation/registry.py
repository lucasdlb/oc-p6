"""Registry for table-specific imputers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from credit_risk.data.imputation.base import TableImputer
from credit_risk.data.imputation.raw import RawImputer


class ImputationRegistry:
    """Registry for looking up table-specific imputers."""

    _registry: dict[str, type[TableImputer]] = {}

    @classmethod
    def get_imputer(cls, table: str, method: str = "default") -> TableImputer:
        """Get imputer instance for a table.

        Args:
            table: Table name (e.g., "bureau", "application")
            method: Imputation method ("default" or "raw")

        Returns:
            TableImputer instance
        """
        cls._ensure_initialized()
        if method == "raw":
            return RawImputer()
        if table not in cls._registry:
            registered = list(cls._registry.keys())
            raise KeyError(
                f"No imputer registered for table '{table}'. Did you mean one of: {registered}?"
            )
        return cls._registry[table]()

    @classmethod
    def available_tables(cls) -> list[str]:
        """Return list of registered table names.

        Returns:
            List of table names with registered imputers
        """
        cls._ensure_initialized()
        return list(cls._registry.keys())

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._registry:
            cls._register_defaults()

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default imputers for all tables."""
        from credit_risk.data.imputation.application import ApplicationImputer
        from credit_risk.data.imputation.bureau import BureauImputer
        from credit_risk.data.imputation.bureau_balance import BureauBalanceImputer
        from credit_risk.data.imputation.credit_card import CreditCardImputer
        from credit_risk.data.imputation.installments import InstallmentsImputer
        from credit_risk.data.imputation.pos_cash import POSCashImputer
        from credit_risk.data.imputation.previous_application import PreviousApplicationImputer

        cls._registry["application"] = ApplicationImputer
        cls._registry["bureau"] = BureauImputer
        cls._registry["bureau_balance"] = BureauBalanceImputer
        cls._registry["previous_application"] = PreviousApplicationImputer
        cls._registry["pos_cash_balance"] = POSCashImputer
        cls._registry["installments_payments"] = InstallmentsImputer
        cls._registry["credit_card_balance"] = CreditCardImputer

    @classmethod
    def register(cls, table: str, imputer: type[TableImputer]) -> None:
        """Register a custom imputer for a table.

        Args:
            table: Table name (e.g., "bureau", "application")
            imputer: Imputer class to register
        """
        cls._ensure_initialized()
        cls._registry[table] = imputer

    @classmethod
    @contextmanager
    def override(cls, table: str, imputer: type[TableImputer]) -> Generator[None, None, None]:
        """Temporarily override an imputer for a table.

        Useful for experiments and testing.

        Args:
            table: Table name to override
            imputer: Imputer class to use temporarily

        Yields:
            None
        """
        cls._ensure_initialized()
        previous = cls._registry.get(table)
        cls._registry[table] = imputer
        try:
            yield
        finally:
            if previous is None:
                cls._registry.pop(table, None)
            else:
                cls._registry[table] = previous
