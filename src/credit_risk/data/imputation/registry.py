"""Registry for table-specific imputers."""

from __future__ import annotations

from credit_risk.data.registry import Registry


class ImputationRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk.data.base import NoOpStep
        from credit_risk.data.imputation.application import ApplicationImputer
        from credit_risk.data.imputation.bureau import BureauImputer
        from credit_risk.data.imputation.bureau_balance import BureauBalanceImputer
        from credit_risk.data.imputation.credit_card import CreditCardImputer
        from credit_risk.data.imputation.installments import InstallmentsImputer
        from credit_risk.data.imputation.pos_cash import POSCashImputer
        from credit_risk.data.imputation.previous_application import (
            PreviousApplicationImputer,
        )

        cls._registry["ApplicationImputer"] = ApplicationImputer
        cls._registry["BureauImputer"] = BureauImputer
        cls._registry["BureauBalanceImputer"] = BureauBalanceImputer
        cls._registry["CreditCardImputer"] = CreditCardImputer
        cls._registry["InstallmentsImputer"] = InstallmentsImputer
        cls._registry["POSCashImputer"] = POSCashImputer
        cls._registry["PreviousApplicationImputer"] = PreviousApplicationImputer
        cls._registry["NoOpStep"] = NoOpStep
