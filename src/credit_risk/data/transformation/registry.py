"""Registry for table-specific transformers."""

from __future__ import annotations

from credit_risk.data.registry import Registry


class TransformerRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk.data.base import NoOpStep
        from credit_risk.data.transformation.application import ApplicationTransformer
        from credit_risk.data.transformation.bureau import BureauTransformer
        from credit_risk.data.transformation.bureau_balance import BureauBalanceTransformer
        from credit_risk.data.transformation.credit_card import CreditCardTransformer
        from credit_risk.data.transformation.installments import InstallmentsTransformer
        from credit_risk.data.transformation.pos_cash import POSCashTransformer
        from credit_risk.data.transformation.previous_application import (
            PreviousApplicationTransformer,
        )

        cls._registry["ApplicationTransformer"] = ApplicationTransformer
        cls._registry["BureauTransformer"] = BureauTransformer
        cls._registry["BureauBalanceTransformer"] = BureauBalanceTransformer
        cls._registry["CreditCardTransformer"] = CreditCardTransformer
        cls._registry["InstallmentsTransformer"] = InstallmentsTransformer
        cls._registry["NoOpStep"] = NoOpStep
        cls._registry["POSCashTransformer"] = POSCashTransformer
        cls._registry["PreviousApplicationTransformer"] = PreviousApplicationTransformer
