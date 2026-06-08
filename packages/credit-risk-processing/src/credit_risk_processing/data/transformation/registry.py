"""Registry for table-specific transformers."""

from __future__ import annotations

from credit_risk_processing.data.registry import Registry


class TransformerRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk_processing.data.base import NoOpStep
        from credit_risk_processing.data.transformation.application import ApplicationTransformer
        from credit_risk_processing.data.transformation.bureau import BureauTransformer
        from credit_risk_processing.data.transformation.bureau_balance import BureauBalanceTransformer
        from credit_risk_processing.data.transformation.credit_card import CreditCardBalanceTransformer
        from credit_risk_processing.data.transformation.cross import CrossTableTransformer
        from credit_risk_processing.data.transformation.installments import InstallmentsTransformer
        from credit_risk_processing.data.transformation.pos_cash import POSCashBalanceTransformer
        from credit_risk_processing.data.transformation.previous_application import (
            PreviousApplicationTransformer,
        )

        cls._registry["ApplicationTransformer"] = ApplicationTransformer
        cls._registry["BureauTransformer"] = BureauTransformer
        cls._registry["BureauBalanceTransformer"] = BureauBalanceTransformer
        cls._registry["CreditCardBalanceTransformer"] = CreditCardBalanceTransformer
        cls._registry["CrossTableTransformer"] = CrossTableTransformer
        cls._registry["InstallmentsTransformer"] = InstallmentsTransformer
        cls._registry["NoOpStep"] = NoOpStep
        cls._registry["POSCashBalanceTransformer"] = POSCashBalanceTransformer
        cls._registry["PreviousApplicationTransformer"] = PreviousApplicationTransformer
