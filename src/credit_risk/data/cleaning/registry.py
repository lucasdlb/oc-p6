"""Registry for table-specific cleaners."""

from __future__ import annotations

from credit_risk.data.registry import Registry


class CleaningRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk.data.base import NoOpStep
        from credit_risk.data.cleaning.application import ApplicationCleaner
        from credit_risk.data.cleaning.bureau import BureauCleaner
        from credit_risk.data.cleaning.bureau_balance import BureauBalanceCleaner
        from credit_risk.data.cleaning.credit_card import CreditCardBalanceCleaner
        from credit_risk.data.cleaning.installments import InstallmentsCleaner
        from credit_risk.data.cleaning.pos_cash import POSCashBalanceCleaner
        from credit_risk.data.cleaning.previous_application import PreviousApplicationCleaner

        cls._registry["ApplicationCleaner"] = ApplicationCleaner
        cls._registry["BureauCleaner"] = BureauCleaner
        cls._registry["BureauBalanceCleaner"] = BureauBalanceCleaner
        cls._registry["CreditCardBalanceCleaner"] = CreditCardBalanceCleaner
        cls._registry["InstallmentsCleaner"] = InstallmentsCleaner
        cls._registry["POSCashBalanceCleaner"] = POSCashBalanceCleaner
        cls._registry["PreviousApplicationCleaner"] = PreviousApplicationCleaner
        cls._registry["NoOpStep"] = NoOpStep
