"""Bureau table aggregator."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame, LazyFrame

from credit_risk.config import FeaturesConfig
from credit_risk.data.aggregation._common import (
    AGG_METHODS,
    _categorical_diversity_exprs,
    _get_agg_exprs,
)
from credit_risk.data.base import StatelessStep


class BureauAggregator(StatelessStep):
    """Aggregator for bureau table.

    Domain features:
    - Credit type and status distribution
    - Active vs closed credit ratios
    - Days since credit and overdue statistics
    - Annuity amount patterns
    - Categorical diversity (top_freq, entropy) for CREDIT_ACTIVE, CREDIT_TYPE
    """

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    @override
    def fit(self, X: DataFrame, y=None) -> BureauAggregator:
        self._methods = AGG_METHODS["default"]
        return self

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        lf = X.lazy()
        schema = dict(X.schema)

        if not hasattr(self, "_agg_cols"):
            self._agg_cols = [
                c for c in X.columns if c not in {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}
            ]
            self._methods = list(AGG_METHODS["default"])

        agg_exprs = _get_agg_exprs(self._agg_cols, schema, "bureau_", self._methods)
        agg_exprs.append(pl.col("SK_ID_CURR").count().alias("bureau_n_records"))

        result = lf.group_by("SK_ID_CURR").agg(*agg_exprs)
        result = result.join(self._credit_status_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._credit_type_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._days_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._categorical_diversity(lf), on="SK_ID_CURR", how="left")

        return result.collect()

    def _credit_status_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            (pl.col("CREDIT_ACTIVE") == "Active").sum().alias("bureau_active_count"),
            (pl.col("CREDIT_ACTIVE") == "Closed").sum().alias("bureau_closed_count"),
            ((pl.col("CREDIT_ACTIVE") == "Active").sum() / (pl.len() + 1)).alias(
                "bureau_active_ratio"
            ),
            (pl.col("AMT_CREDIT_SUM_OVERDUE") > 0).sum().alias("bureau_overdue_count"),
        )

    def _credit_type_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            (pl.col("CREDIT_TYPE") == "Consumer credit").sum().alias("bureau_consumer_count"),
            (pl.col("CREDIT_TYPE") == "Car credit").sum().alias("bureau_car_count"),
            (pl.col("CREDIT_TYPE") == "Microcredit").sum().alias("bureau_micro_count"),
            pl.col("CREDIT_TYPE").n_unique().alias("bureau_credit_type_nunique"),
        )

    def _days_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            pl.col("DAYS_CREDIT").min().alias("bureau_earliest_credit_days"),
            pl.col("DAYS_CREDIT").max().alias("bureau_latest_credit_days"),
            (pl.col("DAYS_CREDIT").max() - pl.col("DAYS_CREDIT").min()).alias(
                "bureau_credit_span_days"
            ),
            pl.col("AMT_ANNUITY").mean().alias("bureau_annuity_mean"),
            pl.col("AMT_ANNUITY").max().alias("bureau_annuity_max"),
        )

    def _categorical_diversity(self, lf: LazyFrame) -> LazyFrame:
        ca_diversity = _categorical_diversity_exprs(lf, "SK_ID_CURR", "CREDIT_ACTIVE", "bureau_")
        ct_diversity = _categorical_diversity_exprs(lf, "SK_ID_CURR", "CREDIT_TYPE", "bureau_")
        return ca_diversity.join(ct_diversity, on="SK_ID_CURR", how="outer")
