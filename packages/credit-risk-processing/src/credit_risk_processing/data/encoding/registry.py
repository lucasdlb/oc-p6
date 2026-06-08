"""Registry for table-specific encoders."""

from __future__ import annotations

from credit_risk_processing.data.encoding.onehot import PolarsTargetEncoder
from credit_risk_processing.data.registry import Registry


class EncodingRegistry(Registry):
    _registry: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        from credit_risk_processing.data.base import NoOpStep
        from credit_risk_processing.data.encoding.categorical import CategoricalEncoder
        from credit_risk_processing.data.encoding.onehot import PolarsOneHotEncoder

        cls._registry["CategoricalEncoder"] = CategoricalEncoder
        cls._registry["NoOpStep"] = NoOpStep
        cls._registry["PolarsOneHotEncoder"] = PolarsOneHotEncoder
        cls._registry["PolarsTargetEncoder"] = PolarsTargetEncoder
