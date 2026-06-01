"""End-to-end data processing pipeline.

ProcessingPipeline builds a sklearn Pipeline from TableConfig.
Each Pipeline runs: clean → impute → aggregate → transform → encode.

The build() method returns a sklearn Pipeline, which handles fit/transform
chaining automatically and supports clone() for clean re-instantiation.
"""

from __future__ import annotations

from credit_risk_processing.data.aggregation.registry import AggregatorRegistry
from credit_risk_processing.data.base import SchemaEnforcer
from credit_risk_processing.data.cleaning.registry import CleaningRegistry
from credit_risk_processing.data.encoding.registry import EncodingRegistry
from credit_risk_processing.data.imputation.registry import ImputationRegistry
from credit_risk_processing.data.transformation.registry import TransformerRegistry
from sklearn.pipeline import Pipeline

from credit_risk.config import TableConfig


class ProcessingPipeline:
    """Builds an sklearn Pipeline from TableConfig.

    Usage::

        pipe = ProcessingPipeline(cfg.data.bureau).build()
        X_train = pipe.fit_transform(df_train)
        X_val    = pipe.transform(df_val)
    """

    def __init__(self, table_cfg: TableConfig) -> None:
        self.table_cfg = table_cfg

    def build(self) -> Pipeline:
        """Build the sklearn Pipeline."""
        return Pipeline(
            [
                ("cleaner", CleaningRegistry.get(self.table_cfg.cleaner)()),
                ("imputer", ImputationRegistry.get(self.table_cfg.imputer)()),
                ("aggregator", AggregatorRegistry.get(self.table_cfg.aggregator)()),
                ("transformer", TransformerRegistry.get(self.table_cfg.transformer)()),
                ("encoder", EncodingRegistry.get(self.table_cfg.encoder)()),
                ("schema", SchemaEnforcer()),
            ]
        )
