"""End-to-end data processing pipeline.

ProcessingPipeline is a per-table composite that runs all five steps in
sequence: clean → impute → aggregate → transform → encode.

Each step is resolved from its registry using the class names in
``TableConfig``.  Tables that don't need a particular step should
configure a NoOp class (e.g. ``NoOpAggregator``, ``NoOpEncoder``).

Follows the sklearn-style ``fit`` / ``transform`` / ``fit_transform``
protocol so the pipeline can be used on training data (fit_transform)
and later on test data (transform only).
"""

from __future__ import annotations

from typing import Self, override

from polars import DataFrame

from credit_risk.config import TableConfig
from credit_risk.data.aggregation.registry import AggregatorRegistry
from credit_risk.data.base import ProcessingStep
from credit_risk.data.cleaning.registry import CleaningRegistry
from credit_risk.data.encoding.registry import EncodingRegistry
from credit_risk.data.imputation.registry import ImputationRegistry
from credit_risk.data.transformation.registry import TransformerRegistry


class ProcessingPipeline(ProcessingStep):
    """Per-table processing pipeline: clean → impute → aggregate → transform → encode.

    Usage::

        pipe = ProcessingPipeline(cfg.data.bureau)
        df_train = pipe.fit_transform(df_train)
        df_test  = pipe.transform(df_test)
    """

    def __init__(self, table_cfg: TableConfig) -> None:
        self._cleaner = CleaningRegistry.get(table_cfg.cleaner)()
        self._imputer = ImputationRegistry.get(table_cfg.imputer)()
        self._aggregator = AggregatorRegistry.get(table_cfg.aggregator)()
        self._transformer = TransformerRegistry.get(table_cfg.transformer)()
        self._encoder = EncodingRegistry.get(table_cfg.encoder)()

    @override
    def fit(self, df: DataFrame) -> Self:
        self._cleaner.fit(df)
        df = self._cleaner.transform(df)
        self._imputer.fit(df)
        df = self._imputer.transform(df)
        self._aggregator.fit(df)
        df = self._aggregator.transform(df)
        self._transformer.fit(df)
        df = self._transformer.transform(df)
        self._encoder.fit(df)
        return self

    @override
    def transform(self, df: DataFrame) -> DataFrame:
        df = self._cleaner.transform(df)
        df = self._imputer.transform(df)
        df = self._aggregator.transform(df)
        df = self._transformer.transform(df)
        df = self._encoder.transform(df)
        return df
