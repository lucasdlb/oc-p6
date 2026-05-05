"""Previous application table aggregator."""

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


class PreviousApplicationAggregator(StatelessStep):
    """Aggregator for previous_application table.

    Domain features:
    - Approval/refusal/cancellation counts and rates
    - Credit gap ratios for approved applications
    - Recency features (last 1-year window)
    - Refused application amount statistics
    - Client type diversity and distribution
    - Yield group ordinal aggregation
    - Channel and product diversity
    - Categorical diversity (top_freq, entropy) for NAME_CLIENT_TYPE
    """

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    @override
    def fit(self, X: DataFrame, y=None) -> PreviousApplicationAggregator:
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

        agg_exprs = _get_agg_exprs(self._agg_cols, schema, "prev_", self._methods)
        agg_exprs.append(pl.len().alias("prev_n_records"))

        result = lf.group_by("SK_ID_CURR").agg(*agg_exprs)
        result = result.join(self._approval_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._credit_gap_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._recency_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._refused_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._client_type_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._yield_group_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._diversity_features(lf), on="SK_ID_CURR", how="left")
        result = result.join(self._categorical_diversity(lf), on="SK_ID_CURR", how="left")

        return result.collect()

    def _approval_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            (pl.col("NAME_CONTRACT_STATUS") == "Approved").sum().alias("prev_approved_count"),
            (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("prev_refused_count"),
            (pl.col("NAME_CONTRACT_STATUS") == "Canceled").sum().alias("prev_canceled_count"),
            (pl.col("NAME_CONTRACT_STATUS") == "Unused offer").sum().alias("prev_unused_count"),
            ((pl.col("NAME_CONTRACT_STATUS") == "Approved").sum() / (pl.len() + 1)).alias(
                "prev_approval_rate"
            ),
            ((pl.col("NAME_CONTRACT_STATUS") == "Refused").sum() / (pl.len() + 1)).alias(
                "prev_refusal_rate"
            ),
        )

    def _credit_gap_features(self, lf: LazyFrame) -> LazyFrame:
        approved = lf.filter(pl.col("NAME_CONTRACT_STATUS") == "Approved")
        return approved.group_by("SK_ID_CURR").agg(
            (pl.col("AMT_CREDIT") / (pl.col("AMT_APPLICATION") + 1))
            .mean()
            .alias("prev_credit_to_application_ratio"),
            (pl.col("AMT_CREDIT") - pl.col("AMT_APPLICATION")).mean().alias("prev_credit_gap_mean"),
            pl.col("AMT_DOWN_PAYMENT").mean().alias("prev_down_payment_mean"),
            pl.col("RATE_DOWN_PAYMENT").mean().alias("prev_rate_down_payment_mean"),
        )

    def _recency_features(self, lf: LazyFrame) -> LazyFrame:
        recent = lf.filter(pl.col("DAYS_DECISION") >= -365)
        return recent.group_by("SK_ID_CURR").agg(
            pl.len().alias("prev_n_applications_1y"),
            (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("prev_refused_count_1y"),
            (pl.col("NAME_CONTRACT_STATUS") == "Approved").sum().alias("prev_approved_count_1y"),
            pl.col("AMT_CREDIT").mean().alias("prev_amt_credit_mean_1y"),
        )

    def _refused_features(self, lf: LazyFrame) -> LazyFrame:
        refused = lf.filter(pl.col("NAME_CONTRACT_STATUS") == "Refused")
        return refused.group_by("SK_ID_CURR").agg(
            pl.col("AMT_APPLICATION").sum().alias("prev_refused_amt_sum"),
            pl.col("AMT_APPLICATION").mean().alias("prev_refused_amt_mean"),
            pl.col("AMT_APPLICATION").max().alias("prev_refused_amt_max"),
        )

    def _client_type_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            (pl.col("NAME_CLIENT_TYPE") == "New").sum().alias("prev_client_new_count"),
            (pl.col("NAME_CLIENT_TYPE") == "Repeater").sum().alias("prev_client_repeater_count"),
            (pl.col("NAME_CLIENT_TYPE") == "Refreshed").sum().alias("prev_client_refreshed_count"),
            ((pl.col("NAME_CLIENT_TYPE") == "New").sum() / (pl.len() + 1)).alias(
                "prev_client_new_rate"
            ),
        )

    def _yield_group_features(self, lf: LazyFrame) -> LazyFrame:
        yield_map = {"low_normal": 1, "low_action": 2, "middle": 3, "high": 4}
        return lf.group_by("SK_ID_CURR").agg(
            pl.col("NAME_YIELD_GROUP")
            .replace(yield_map)
            .cast(pl.Float64)
            .mean()
            .alias("prev_yield_group_mean"),
            (pl.col("NAME_YIELD_GROUP") == "high").sum().alias("prev_yield_high_count"),
            (pl.col("NAME_YIELD_GROUP") == "low_normal").sum().alias("prev_yield_low_count"),
        )

    def _diversity_features(self, lf: LazyFrame) -> LazyFrame:
        return lf.group_by("SK_ID_CURR").agg(
            pl.col("CHANNEL_TYPE").n_unique().alias("prev_channel_nunique"),
            pl.col("PRODUCT_COMBINATION").n_unique().alias("prev_product_comb_nunique"),
            pl.col("NAME_CONTRACT_TYPE").n_unique().alias("prev_contract_type_nunique"),
        )

    def _categorical_diversity(self, lf: LazyFrame) -> LazyFrame:
        return _categorical_diversity_exprs(lf, "SK_ID_CURR", "NAME_CLIENT_TYPE", "prev_")
