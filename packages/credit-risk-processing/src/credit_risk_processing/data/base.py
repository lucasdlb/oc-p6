"""Abstract base class for all pipeline steps.

**ProcessingStep** — sklearn-compatible base (BaseEstimator + TransformerMixin).

**StatelessStep** — convenience base for steps that don't learn from data.

**NoOpStep** — stateless identity transform.

Every step follows fit → transform → fit_transform (from TransformerMixin):
- ``fit(X, y=None)`` learns parameters from training data
- ``transform(X, y=None)`` applies the transformation
- ``fit_transform(X, y=None)`` does both in one pass (provided by TransformerMixin)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Self, override

import polars as pl
from polars import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class ProcessingStep(BaseEstimator, TransformerMixin):
    """Base for all data-processing steps.

    Inherits from sklearn's BaseEstimator and TransformerMixin:
    - BaseEstimator  → get_params / set_params (required for clone(), Pipeline, GridSearch)
    - TransformerMixin → fit_transform default implementation

    Subclasses should override __sklearn_is_fitted__ if they set trailing-
    underscore fitted attributes (the default implementation below returns
    False, deferring to check_is_fitted which looks for those attributes).
    """

    @abstractmethod
    def fit(self, X: DataFrame, y=None) -> Self:
        """Learn parameters from training data.

        y is accepted but ignored by default — required for sklearn Pipeline
        compatibility when a step receives y (e.g. TargetEncoder steps).
        """

    @abstractmethod
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """Apply the step's transformation to X."""


class StatelessStep(ProcessingStep):
    """Processing step that holds no learned state.

    fit() is a no-op — the step applies the same transformation
    regardless of the data it was fitted on.

    Concretes must implement transform.
    """

    @override
    def fit(self, X: DataFrame, y=None) -> Self:

        return self


class NoOpStep(StatelessStep):
    """Identity processing step — passes data through unchanged."""

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return X


class SchemaEnforcer(ProcessingStep):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        return X.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64).alias("SK_ID_CURR"))
