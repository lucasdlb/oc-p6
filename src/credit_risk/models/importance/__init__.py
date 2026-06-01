"""Importance strategies registry."""

from __future__ import annotations

from typing import Type

from credit_risk.models.importance.base import BaseImportanceStrategy
from credit_risk.models.importance.inner import InnerImportance
from credit_risk.models.importance.permutation import PermutationImportance
from credit_risk.models.importance.shap import SHAPImportance
from credit_risk.models.importance.statistical import StatisticalImportance

REGISTRY: dict[str, Type[BaseImportanceStrategy]] = {
    "inner": InnerImportance,
    "statistical": StatisticalImportance,
    "permutation": PermutationImportance,
    "shap": SHAPImportance,
}


def get_importance_class(name: str) -> Type[BaseImportanceStrategy]:
    """Get importance class by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown importance method: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


__all__ = [
    "BaseImportanceStrategy",
    "InnerImportance",
    "StatisticalImportance",
    "PermutationImportance",
    "SHAPImportance",
    "REGISTRY",
    "get_importance_class",
]
