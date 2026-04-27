"""Registry for table-specific aggregators."""

from __future__ import annotations

from credit_risk.data.registry import Registry


class AggregatorRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk.data.aggregation.bureau import BureauAggregator
        from credit_risk.data.aggregation.bureau_balance import BureauBalanceAggregator
        from credit_risk.data.aggregation.credit_card import CreditCardAggregator
        from credit_risk.data.aggregation.installments import InstallmentsAggregator
        from credit_risk.data.aggregation.pos_cash import POSCashAggregator
        from credit_risk.data.aggregation.previous_application import (
            PreviousApplicationAggregator,
        )
        from credit_risk.data.base import NoOpStep

        cls._registry["BureauAggregator"] = BureauAggregator
        cls._registry["BureauBalanceAggregator"] = BureauBalanceAggregator
        cls._registry["CreditCardAggregator"] = CreditCardAggregator
        cls._registry["InstallmentsAggregator"] = InstallmentsAggregator
        cls._registry["NoOpStep"] = NoOpStep
        cls._registry["POSCashAggregator"] = POSCashAggregator
        cls._registry["PreviousApplicationAggregator"] = PreviousApplicationAggregator
