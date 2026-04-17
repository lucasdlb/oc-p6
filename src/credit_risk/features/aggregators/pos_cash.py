"""POS CASH balance table aggregator."""

from __future__ import annotations

import polars as pl
from polars import LazyFrame

from credit_risk.config import FeaturesConfig
from credit_risk.features.aggregators.base import TableAggregator

AGG_METHODS = {
    "minimal": ["mean", "max", "count"],
    "default": ["mean", "sum", "min", "max", "std", "count"],
    "detailed": ["mean", "sum", "min", "max", "std", "count", "quantile"],
}


def _get_agg_exprs(
    columns: list[str], numeric_cols: set[str], prefix: str = "", methods: list[str] | None = None
) -> list[pl.Expr]:
    if methods is None:
        methods = list(AGG_METHODS["default"])
    exprs = []
    for col in columns:
        if col in numeric_cols:
            for func in methods:
                if func == "quantile":
                    exprs.append(pl.col(col).quantile(0.5).alias(f"{prefix}{col}_quantile"))
                else:
                    exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
        else:
            for func in ["first", "count"]:
                exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
    return exprs


def _get_numeric_cols(lf: LazyFrame, columns: list[str]) -> set[str]:
    schema = lf.collect_schema()
    return {col for col in columns if col in schema and schema[col] not in (pl.String, pl.Boolean)}


class POSCashAggregator(TableAggregator):
    """Aggregator for POS_CASH_balance table.

    Key signals in this table:
    - SK_DPD / SK_DPD_DEF: days past due — direct default signal
    - CNT_INSTALMENT_FUTURE: remaining instalments — completion proxy
    - NAME_CONTRACT_STATUS: active vs completed vs cancelled
    - MONTHS_BALANCE: recency — recent months matter more than old ones
    """

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    @classmethod
    def load_link(cls) -> LazyFrame | None:
        return None

    def aggregate(self, df: LazyFrame, method: str = "default") -> LazyFrame:
        agg_cols = self.config.pos_cash_agg_features
        numeric_cols = _get_numeric_cols(df, agg_cols)
        methods = AGG_METHODS.get(method, AGG_METHODS[method])

        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "pos_", methods)
        agg_exprs.append(pl.len().alias("pos_n_records"))

        result = df.group_by("SK_ID_CURR").agg(*agg_exprs)

        # --- high-value derived aggregations ---
        result = result.join(self._dpd_features(df), on="SK_ID_CURR", how="left")
        result = result.join(self._recency_features(df), on="SK_ID_CURR", how="left")
        result = result.join(self._status_features(df), on="SK_ID_CURR", how="left")

        return result

    def _dpd_features(self, df: LazyFrame) -> LazyFrame:
        """DPD-based features — most direct default signal in this table."""
        return df.group_by("SK_ID_CURR").agg(
            pl.col("SK_DPD").max().alias("pos_dpd_max"),
            pl.col("SK_DPD").mean().alias("pos_dpd_mean"),
            (pl.col("SK_DPD") > 0).sum().alias("pos_dpd_months_count"),
            pl.col("SK_DPD_DEF").max().alias("pos_dpd_def_max"),
            (pl.col("SK_DPD_DEF") > 0).sum().alias("pos_dpd_def_months_count"),
        )

    def _recency_features(self, df: LazyFrame) -> LazyFrame:
        """Features computed on last 12 months only — recent behaviour matters more."""
        recent = df.filter(pl.col("MONTHS_BALANCE") >= -12)
        return recent.group_by("SK_ID_CURR").agg(
            pl.col("SK_DPD").max().alias("pos_dpd_max_12m"),
            pl.col("SK_DPD").mean().alias("pos_dpd_mean_12m"),
            (pl.col("SK_DPD") > 0).sum().alias("pos_dpd_months_count_12m"),
            pl.col("CNT_INSTALMENT_FUTURE").min().alias("pos_instalments_remaining_12m"),
        )

    def _status_features(self, df: LazyFrame) -> LazyFrame:
        """Contract status aggregations — completion rate is a strong signal."""
        return df.group_by("SK_ID_CURR").agg(
            (pl.col("NAME_CONTRACT_STATUS") == "Completed").sum().alias("pos_completed_count"),
            (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("pos_active_count"),
            ((pl.col("NAME_CONTRACT_STATUS") == "Completed").sum() / (pl.len() + 1)).alias(
                "pos_completion_rate"
            ),
        )
