"""Previous application table aggregator."""

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


class PreviousApplicationAggregator(TableAggregator):
    """Aggregator for previous_application table.

    Key signals in this table:
    - NAME_CONTRACT_STATUS: approved/refused ratio — top feature across solutions
    - AMT_CREDIT vs AMT_APPLICATION: how much was actually granted vs requested
    - DAYS_DECISION: recency of previous applications
    - NAME_CONTRACT_TYPE: cash vs revolving vs consumer credit mix
    - RATE_DOWN_PAYMENT: financial commitment signal
    """

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import load_config

        self.config = config or load_config().data.features

    @classmethod
    def load_link(cls) -> LazyFrame | None:
        return None

    def aggregate(self, df: LazyFrame, method: str = "default") -> LazyFrame:
        agg_cols = self.config.previous_app_agg_features
        numeric_cols = _get_numeric_cols(df, agg_cols)
        methods = AGG_METHODS.get(method, AGG_METHODS[method])

        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "prev_", methods)
        agg_exprs.append(pl.len().alias("prev_n_records"))

        result = df.group_by("SK_ID_CURR").agg(*agg_exprs)
        result = result.join(self._approval_features(df), on="SK_ID_CURR", how="left")
        result = result.join(self._credit_gap_features(df), on="SK_ID_CURR", how="left")
        result = result.join(self._recency_features(df), on="SK_ID_CURR", how="left")
        result = result.join(self._refused_features(df), on="SK_ID_CURR", how="left")

        return result

    def _approval_features(self, df: LazyFrame) -> LazyFrame:
        """Approval/refusal ratios — consistently top features in published solutions."""
        return df.group_by("SK_ID_CURR").agg(
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

    def _credit_gap_features(self, df: LazyFrame) -> LazyFrame:
        """Gap between requested and granted credit — measures lender risk perception."""
        approved = df.filter(pl.col("NAME_CONTRACT_STATUS") == "Approved")
        return approved.group_by("SK_ID_CURR").agg(
            (pl.col("AMT_CREDIT") / (pl.col("AMT_APPLICATION") + 1))
            .mean()
            .alias("prev_credit_to_application_ratio"),
            (pl.col("AMT_CREDIT") - pl.col("AMT_APPLICATION")).mean().alias("prev_credit_gap_mean"),
            pl.col("AMT_DOWN_PAYMENT").mean().alias("prev_down_payment_mean"),
            pl.col("RATE_DOWN_PAYMENT").mean().alias("prev_rate_down_payment_mean"),
        )

    def _recency_features(self, df: LazyFrame) -> LazyFrame:
        """Most recent application features — recent behaviour outweighs old history."""
        recent = df.filter(pl.col("DAYS_DECISION") >= -365)
        return recent.group_by("SK_ID_CURR").agg(
            pl.len().alias("prev_n_applications_1y"),
            (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("prev_refused_count_1y"),
            (pl.col("NAME_CONTRACT_STATUS") == "Approved").sum().alias("prev_approved_count_1y"),
            pl.col("AMT_CREDIT").mean().alias("prev_amt_credit_mean_1y"),
        )

    def _refused_features(self, df: LazyFrame) -> LazyFrame:
        """Features computed only on refused applications.

        How much credit was refused and for what purpose — signals
        whether lenders elsewhere perceived this applicant as risky.
        """
        refused = df.filter(pl.col("NAME_CONTRACT_STATUS") == "Refused")
        return refused.group_by("SK_ID_CURR").agg(
            pl.col("AMT_APPLICATION").sum().alias("prev_refused_amt_sum"),
            pl.col("AMT_APPLICATION").mean().alias("prev_refused_amt_mean"),
            pl.col("AMT_APPLICATION").max().alias("prev_refused_amt_max"),
        )
