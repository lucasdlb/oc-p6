"""Registry for table-specific aggregators."""

from __future__ import annotations

from credit_risk.data.registry import Registry


class AggregatorRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk.data.aggregation.bureau import BureauAggregator
        from credit_risk.data.aggregation.bureau_balance import (
            BureauBalanceAggregator,
            DefaultBureauBalanceAggregator,
            DetailedBureauBalanceAggregator,
            MinimalBureauBalanceAggregator,
        )
        from credit_risk.data.aggregation.credit_card import (
            CreditCardBalanceAggregator,
            DefaultCreditCardAggregator,
            DetailedCreditCardAggregator,
            MinimalCreditCardAggregator,
        )
        from credit_risk.data.aggregation.installments import (
            DefaultInstallmentsAggregator,
            DetailedInstallmentsAggregator,
            MinimalInstallmentsAggregator,
        )
        from credit_risk.data.aggregation.pos_cash import POSCashBalanceAggregator
        from credit_risk.data.aggregation.previous_application import (
            PreviousApplicationAggregator,
        )
        from credit_risk.data.base import NoOpStep

        cls._registry["BureauAggregator"] = BureauAggregator
        cls._registry["MinimalBureauBalanceAggregator"] = MinimalBureauBalanceAggregator
        cls._registry["DefaultBureauBalanceAggregator"] = DefaultBureauBalanceAggregator
        cls._registry["DetailedBureauBalanceAggregator"] = DetailedBureauBalanceAggregator
        cls._registry["BureauBalanceAggregator"] = BureauBalanceAggregator
        cls._registry["CreditCardBalanceAggregator"] = CreditCardBalanceAggregator
        cls._registry["MinimalCreditCardAggregator"] = MinimalCreditCardAggregator
        cls._registry["DefaultCreditCardAggregator"] = DefaultCreditCardAggregator
        cls._registry["DetailedCreditCardAggregator"] = DetailedCreditCardAggregator
        cls._registry["MinimalInstallmentsAggregator"] = MinimalInstallmentsAggregator
        cls._registry["DefaultInstallmentsAggregator"] = DefaultInstallmentsAggregator
        cls._registry["DetailedInstallmentsAggregator"] = DetailedInstallmentsAggregator
        cls._registry["InstallmentsAggregator"] = DefaultInstallmentsAggregator
        cls._registry["NoOpStep"] = NoOpStep
        cls._registry["POSCashBalanceAggregator"] = POSCashBalanceAggregator
        cls._registry["PreviousApplicationAggregator"] = PreviousApplicationAggregator
