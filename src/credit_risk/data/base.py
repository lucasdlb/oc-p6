"""Abstract base class for all pipeline steps.

**ProcessingStep** — the single abstract base for every processing step.

**StatelessStep** — convenience base for steps that don't learn from data.

Every step follows fit → transform → fit_transform:
- ``fit(df)`` learns parameters from training data (override for stateful steps)
- ``transform(df)`` applies the transformation (always required)
- ``fit_transform(df)`` does both in one pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

from polars import DataFrame


class ProcessingStep(ABC):
    """Base for all data-processing steps.

    Concretes must implement both ``fit`` and ``transform``.
    """

    @abstractmethod
    def fit(self, df: DataFrame) -> Self:
        """Learn parameters from training data.

        Stateless steps should return ``self`` without modifying anything.
        """

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply the step's transformation to ``df``."""

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Fit and transform in one pass."""
        return self.fit(df).transform(df)


class StatelessStep(ProcessingStep):
    """Processing step that holds no learned state.

    ``fit()`` is a no-op — the step applies the same transformation
    regardless of the data it was fitted on.  Use this base for
    cleaners, transformers, and no-op implementations.

    Concretes must implement ``transform``.

    Example::

        class MyCleaner(StatelessStep):
            @override
            def transform(self, df: DataFrame) -> DataFrame:
                return df
    """

    @override
    def fit(self, df: DataFrame) -> Self:
        return self

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply the stateless transformation."""
        ...


class NoOpStep(StatelessStep):
    """Identity processing step — passes data through unchanged."""

    @override
    def transform(self, df: DataFrame) -> DataFrame:
        return df
