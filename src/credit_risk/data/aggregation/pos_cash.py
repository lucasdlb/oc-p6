"""POS CASH balance table aggregator."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame, LazyFrame

from credit_risk.config import FeaturesConfig
from credit_risk.data.aggregation._common import AGG_METHODS, _get_agg_exprs, _recency_weighted_mean
from credit_risk.data.base import StatelessStep


class POSCashBalanceAggregator(StatelessStep):
    """Aggregator for POS_CASH_balance table."""

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    @override
    def fit(self, X: DataFrame, y=None) -> POSCashBalanceAggregator:
        self._methods = AGG_METHODS["default"]
        return self

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        lf = X.lazy()

        if not hasattr(self, "_agg_cols"):
            self._agg_cols = [
                c for c in X.columns if c not in {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}
            ]
            self._methods = list(AGG_METHODS["default"])

        agg_exprs = _get_agg_exprs(self._agg_cols, dict(X.schema), "pos_", self._methods)
        agg_exprs.append(pl.len().alias("pos_n_records"))

        result = lf.group_by("SK_ID_CURR").agg(*agg_exprs)
        result = result.join(self._dpd_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._recency_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._status_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._demand_risk_features(lf), on="SK_ID_CURR", how="left")

        return result.collect()

    def _dpd_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            pl.col("SK_DPD").max().alias("pos_dpd_max"),
            pl.col("SK_DPD").mean().alias("pos_dpd_mean"),
            (pl.col("SK_DPD") > 0).sum().alias("pos_dpd_months_count"),
            pl.col("SK_DPD_DEF").max().alias("pos_dpd_def_max"),
            (pl.col("SK_DPD_DEF") > 0).sum().alias("pos_dpd_def_months_count"),
        )

    def _recency_features(self, lf: LazyFrame) -> LazyFrame:
        recent = lf.filter(pl.col("MONTHS_BALANCE") >= -12)
        return recent.group_by("SK_ID_CURR").agg(
            pl.col("SK_DPD").max().alias("pos_dpd_max_12m"),
            pl.col("SK_DPD").mean().alias("pos_dpd_mean_12m"),
            (pl.col("SK_DPD") > 0).sum().alias("pos_dpd_months_count_12m"),
            pl.col("CNT_INSTALMENT_FUTURE").min().alias("pos_instalments_remaining_12m"),
            _recency_weighted_mean("SK_DPD", "MONTHS_BALANCE").alias("pos_dpd_mean_recent"),
        )

    def _status_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            (pl.col("NAME_CONTRACT_STATUS") == "Completed").sum().alias("pos_completed_count"),
            (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("pos_active_count"),
            ((pl.col("NAME_CONTRACT_STATUS") == "Completed").sum() / (pl.len() + 1)).alias(
                "pos_completion_rate"
            ),
            (pl.col("NAME_CONTRACT_STATUS") == "Demand").sum().alias("pos_demand_count"),
            ((pl.col("NAME_CONTRACT_STATUS") == "Demand").sum() / (pl.len() + 1)).alias(
                "pos_demand_rate"
            ),
            (pl.col("NAME_CONTRACT_STATUS") == "Signed").sum().alias("pos_signed_count"),
        )

    def _demand_risk_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            ((pl.col("NAME_CONTRACT_STATUS") == "Active") & (pl.col("SK_DPD") > 0))
            .sum()
            .alias("pos_active_dpd_count"),
            ((pl.col("NAME_CONTRACT_STATUS") == "Demand") & (pl.col("SK_DPD") > 0))
            .sum()
            .alias("pos_demand_dpd_count"),
        )
