"""Bureau Balance aggregator — competition-grade feature engineering.

Two-level aggregation:
    bureau_balance rows → SK_ID_BUREAU (per credit)
    SK_ID_BUREAU rows  → SK_ID_CURR   (per applicant)

Three aggregation levels (sweep/debug/competition):
    MinimalBureauBalanceAggregator   : 11 bureau features — fast iteration
    DefaultBureauBalanceAggregator   : 21 bureau features — standard training
    DetailedBureauBalanceAggregator   : 28 bureau features — full feature search

In all levels the curr-level aggregation adds 9 hardcoded derived features:
    bb_loans_dpd_ge2/3/5, bb_n_credits, bb_avg_duration,
    bb_pct_loans_ever_bad, bb_credit_history_len, bb_pct_improving, bb_recent_dpd

Feature groups
--------------
Severity  : DPD distribution, delinquency rates, ever-bad flags
Recency   : last-6m / last-12m windows, most-recent DPD
Trend     : covariance slope, first/second-half comparison, streak analysis
Closure   : STATUS C/0 ratios, closed-early detection
"""

from __future__ import annotations

import polars as pl
from polars import DataFrame, Expr, LazyFrame

from credit_risk.data.aggregation._common import _recency_weighted_mean
from credit_risk.data.base import StatelessStep

STATUS_MAP: dict[str, int] = {
    "C": 0,
    "X": -1,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

STATUS_CLOSED_MAP: dict[str, int] = {
    "C": 1,
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "X": 0,
}

_CURR_DERIVED = [
    "bb_loans_dpd_ge2",
    "bb_loans_dpd_ge3",
    "bb_loans_dpd_ge5",
    "bb_n_credits",
    "bb_avg_duration",
    "bb_pct_loans_ever_bad",
    "bb_credit_history_len",
    "bb_pct_improving",
    "bb_recent_dpd",
]

_BUREAU_FEATURES = {
    "minimal": [
        "bb_max_dpd",
        "bb_delinquency_rate",
        "bb_status_C_count",
        "bb_duration",
        "bb_dpd_trend",
        "bb_improving",
        "bb_ever_dpd_ge2",
        "bb_ever_dpd_ge5",
        "bb_months_balance_min",
        "bb_months_balance_max",
        "bb_status_C_ratio",
    ],
    "default": [
        "bb_max_dpd",
        "bb_mean_dpd",
        "bb_mean_dpd_recent",
        "bb_std_dpd",
        "bb_months_dpd_ge1",
        "bb_months_dpd_ge2",
        "bb_months_dpd_ge3",
        "bb_delinquency_rate",
        "bb_delinquency_rate_ge2",
        "bb_status_C_count",
        "bb_status_C_ratio",
        "bb_status_0_ratio",
        "bb_duration",
        "bb_last6m_max_dpd",
        "bb_last6m_rate",
        "bb_last12m_max_dpd",
        "bb_last12m_rate",
        "bb_months_balance_max",
        "bb_months_balance_min",
        "bb_dpd_trend",
        "bb_ever_dpd_ge2",
        "bb_ever_dpd_ge5",
        "bb_improving",
    ],
    "detailed": [
        "bb_status_0_count",
        "bb_closed_early",
        "bb_first_half_mean_dpd",
        "bb_second_half_mean_dpd",
        "bb_improving",
        "bb_consec_bad_max",
        "bb_consec_good_max",
        "bb_streak_end_dpd",
    ],
}
_BUREAU_FEATURES["detailed"] = list(
    dict.fromkeys(_BUREAU_FEATURES["default"] + _BUREAU_FEATURES["detailed"])
)


def _compute_dpd(df: LazyFrame) -> LazyFrame:
    return (
        df.with_columns(
            pl.col("STATUS").alias("_status_original"),
            pl.col("STATUS").replace(STATUS_MAP).cast(pl.Int8).alias("_dpd_raw"),
            pl.col("STATUS").replace(STATUS_CLOSED_MAP).cast(pl.Int8).alias("bb_is_closed"),
        )
        .sort(["SK_ID_BUREAU", "MONTHS_BALANCE"])
        .with_columns(
            pl.when(pl.col("_dpd_raw") == -1)
            .then(None)
            .otherwise(pl.col("_dpd_raw"))
            .alias("_dpd_for_fill")
        )
        .with_columns(
            pl.col("_dpd_for_fill").forward_fill().over("SK_ID_BUREAU").alias("_dpd_ffilled")
        )
        .with_columns(
            pl.col("_dpd_ffilled").backward_fill().over("SK_ID_BUREAU").alias("_dpd_imputed")
        )
        .with_columns(pl.col("_dpd_imputed").fill_null(0).cast(pl.Int16).alias("DPD"))
        .with_columns(
            (pl.col("DPD") >= 1)
            .cast(pl.Int8)
            .rolling_sum(window_size=6, min_periods=1)
            .over("SK_ID_BUREAU")
            .alias("_rolling_bad"),
            (pl.col("DPD") == 0)
            .cast(pl.Int8)
            .rolling_sum(window_size=6, min_periods=1)
            .over("SK_ID_BUREAU")
            .alias("_rolling_good"),
        )
    )


def _bureau_agg_exprs(features: list[str]) -> list[Expr]:
    n = pl.col("MONTHS_BALANCE").count()
    mid = pl.col("MONTHS_BALANCE").min() / 2

    all_exprs: dict[str, Expr] = {
        "bb_max_dpd": pl.col("DPD").max(),
        "bb_mean_dpd": pl.col("DPD").mean(),
        "bb_std_dpd": pl.col("DPD").std(),
        "bb_mean_dpd_recent": _recency_weighted_mean("DPD", "MONTHS_BALANCE"),
        "bb_months_dpd_ge1": (pl.col("DPD") >= 1).sum(),
        "bb_months_dpd_ge2": (pl.col("DPD") >= 2).sum(),
        "bb_months_dpd_ge3": (pl.col("DPD") >= 3).sum(),
        "bb_delinquency_rate": (pl.col("DPD") >= 1).sum() / n,
        "bb_delinquency_rate_ge2": (pl.col("DPD") >= 2).sum() / n,
        "bb_ever_dpd_ge2": (pl.col("DPD") >= 2).any().cast(pl.Int8),
        "bb_ever_dpd_ge5": (pl.col("DPD") >= 5).any().cast(pl.Int8),
        "bb_duration": n,
        "bb_status_C_count": (pl.col("_status_original") == "C").sum(),
        "bb_status_C_ratio": (pl.col("_status_original") == "C").sum() / n,
        "bb_closed_rate": pl.col("bb_is_closed").mean(),
        "bb_status_0_count": (pl.col("DPD") == 0).sum(),
        "bb_status_0_ratio": (pl.col("DPD") == 0).sum() / n,
        "bb_closed_early": (((pl.col("_status_original") == "C").any() & (n < 24)).cast(pl.Int8)),
        "bb_months_balance_max": pl.col("MONTHS_BALANCE").max(),
        "bb_months_balance_min": pl.col("MONTHS_BALANCE").min(),
        "bb_last6m_max_dpd": pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= -6).max(),
        "bb_last6m_rate": (pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= -6) >= 1).mean(),
        "bb_last12m_max_dpd": pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= -12).max(),
        "bb_last12m_rate": (pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= -12) >= 1).mean(),
        "bb_dpd_trend": pl.cov("MONTHS_BALANCE", "DPD"),
        "bb_first_half_mean_dpd": pl.col("DPD").filter(pl.col("MONTHS_BALANCE") < mid).mean(),
        "bb_second_half_mean_dpd": pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= mid).mean(),
        "bb_improving": (
            (
                pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= mid).mean()
                < pl.col("DPD").filter(pl.col("MONTHS_BALANCE") < mid).mean()
            ).cast(pl.Int8)
        ),
        "bb_slope_reversal": (
            (pl.cov("DPD", "MONTHS_BALANCE") < 0)
            & (pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= mid).mean() > pl.col("DPD").mean())
        ).cast(pl.Int8),
        "bb_consec_bad_max": pl.col("_rolling_bad").max(),
        "bb_consec_good_max": pl.col("_rolling_good").max(),
        "bb_streak_end_dpd": pl.col("DPD").filter(pl.col("MONTHS_BALANCE") >= -3).mean(),
    }

    return [expr.alias(name) for name, expr in all_exprs.items() if name in features]


def _curr_agg_exprs(features: list[str]) -> list[Expr]:
    all_exprs: dict[str, Expr] = {
        "bb_max_dpd": pl.col("bb_max_dpd").max(),
        "bb_mean_dpd": pl.col("bb_mean_dpd").mean(),
        "bb_mean_dpd_recent": pl.col("bb_mean_dpd_recent").mean(),
        "bb_std_dpd": pl.col("bb_std_dpd").mean(),
        "bb_months_dpd_ge1": pl.col("bb_months_dpd_ge1").sum(),
        "bb_months_dpd_ge2": pl.col("bb_months_dpd_ge2").sum(),
        "bb_months_dpd_ge3": pl.col("bb_months_dpd_ge3").sum(),
        "bb_delinquency_rate": pl.col("bb_delinquency_rate").mean(),
        "bb_delinquency_rate_ge2": pl.col("bb_delinquency_rate_ge2").mean(),
        "bb_ever_dpd_ge2": pl.col("bb_ever_dpd_ge2").max(),
        "bb_ever_dpd_ge5": pl.col("bb_ever_dpd_ge5").max(),
        "bb_duration": pl.col("bb_duration").sum(),
        "bb_status_C_count": pl.col("bb_status_C_count").sum(),
        "bb_status_C_ratio": pl.col("bb_status_C_ratio").mean(),
        "bb_closed_rate": pl.col("bb_closed_rate").mean(),
        "bb_status_0_count": pl.col("bb_status_0_count").sum(),
        "bb_status_0_ratio": pl.col("bb_status_0_ratio").mean(),
        "bb_closed_early": pl.col("bb_closed_early").sum(),
        "bb_months_balance_max": pl.col("bb_months_balance_max").max(),
        "bb_months_balance_min": pl.col("bb_months_balance_min").min(),
        "bb_last6m_max_dpd": pl.col("bb_last6m_max_dpd").max(),
        "bb_last6m_rate": pl.col("bb_last6m_rate").mean(),
        "bb_last12m_max_dpd": pl.col("bb_last12m_max_dpd").max(),
        "bb_last12m_rate": pl.col("bb_last12m_rate").mean(),
        "bb_dpd_trend": pl.col("bb_dpd_trend").mean(),
        "bb_first_half_mean_dpd": pl.col("bb_first_half_mean_dpd").mean(),
        "bb_second_half_mean_dpd": pl.col("bb_second_half_mean_dpd").mean(),
        "bb_improving": pl.col("bb_improving").sum(),
        "bb_slope_reversal": pl.col("bb_slope_reversal").max(),
        "bb_consec_bad_max": pl.col("bb_consec_bad_max").max(),
        "bb_consec_good_max": pl.col("bb_consec_good_max").max(),
        "bb_streak_end_dpd": pl.col("bb_streak_end_dpd").mean(),
        "bb_loans_dpd_ge2": (pl.col("bb_max_dpd") >= 2).sum(),
        "bb_loans_dpd_ge3": (pl.col("bb_max_dpd") >= 3).sum(),
        "bb_loans_dpd_ge5": (pl.col("bb_max_dpd") >= 5).sum(),
        "bb_n_credits": pl.col("bb_duration").count(),
        "bb_avg_duration": pl.col("bb_duration").mean(),
        "bb_pct_loans_ever_bad": pl.col("bb_ever_dpd_ge2").mean(),
        "bb_credit_history_len": pl.col("bb_months_balance_min").min().abs(),
        "bb_pct_improving": pl.col("bb_improving").mean(),
        "bb_recent_dpd": pl.col("bb_recent_dpd").max(),
    }

    active = set(features) | set(_CURR_DERIVED)
    return [expr.alias(name) for name, expr in all_exprs.items() if name in active]


class _BaseBureauBalanceAggregator(StatelessStep):
    """Base class for bureau_balance aggregators.

    Daughter classes set BUREAU_FEATURES (list of bureau-level feature names).
    The base handles the two-level aggregation: bureau → curr.
    """

    BUREAU_FEATURES: list[str] = []

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        lf = X.lazy()
        lf = _compute_dpd(lf)

        bb_agg = lf.group_by("SK_ID_BUREAU", "SK_ID_CURR").agg(
            *_bureau_agg_exprs(self.BUREAU_FEATURES)
        )

        bb_agg = bb_agg.join(self._most_recent(lf), on="SK_ID_BUREAU", how="left")

        return bb_agg.group_by("SK_ID_CURR").agg(*_curr_agg_exprs(self.BUREAU_FEATURES)).collect()

    def _most_recent(self, df: LazyFrame) -> LazyFrame:
        return (
            df.filter(
                pl.col("MONTHS_BALANCE") == pl.col("MONTHS_BALANCE").max().over("SK_ID_BUREAU")
            )
            .group_by("SK_ID_BUREAU")
            .agg(pl.col("DPD").max().alias("bb_recent_dpd"))
        )


class MinimalBureauBalanceAggregator(_BaseBureauBalanceAggregator):
    """Minimal bureau_balance aggregator — 11 bureau features for fast iteration."""

    BUREAU_FEATURES = [
        "bb_max_dpd",
        "bb_delinquency_rate",
        "bb_status_C_count",
        "bb_closed_rate",
        "bb_duration",
        "bb_dpd_trend",
        "bb_improving",
        "bb_ever_dpd_ge2",
        "bb_ever_dpd_ge5",
        "bb_months_balance_min",
        "bb_months_balance_max",
        "bb_status_C_ratio",
    ]


class DefaultBureauBalanceAggregator(_BaseBureauBalanceAggregator):
    """Default bureau_balance aggregator — 21 bureau features for standard training."""

    BUREAU_FEATURES = [
        "bb_max_dpd",
        "bb_mean_dpd",
        "bb_mean_dpd_recent",
        "bb_std_dpd",
        "bb_months_dpd_ge1",
        "bb_months_dpd_ge2",
        "bb_months_dpd_ge3",
        "bb_delinquency_rate",
        "bb_delinquency_rate_ge2",
        "bb_status_C_count",
        "bb_status_C_ratio",
        "bb_closed_rate",
        "bb_status_0_ratio",
        "bb_duration",
        "bb_last6m_max_dpd",
        "bb_last6m_rate",
        "bb_last12m_max_dpd",
        "bb_last12m_rate",
        "bb_months_balance_max",
        "bb_months_balance_min",
        "bb_dpd_trend",
        "bb_ever_dpd_ge2",
        "bb_ever_dpd_ge5",
        "bb_improving",
    ]


class DetailedBureauBalanceAggregator(_BaseBureauBalanceAggregator):
    """Detailed bureau_balance aggregator — 28 bureau features for full feature search."""

    BUREAU_FEATURES = [
        "bb_max_dpd",
        "bb_mean_dpd",
        "bb_mean_dpd_recent",
        "bb_std_dpd",
        "bb_months_dpd_ge1",
        "bb_months_dpd_ge2",
        "bb_months_dpd_ge3",
        "bb_delinquency_rate",
        "bb_delinquency_rate_ge2",
        "bb_status_C_count",
        "bb_status_C_ratio",
        "bb_closed_rate",
        "bb_status_0_ratio",
        "bb_duration",
        "bb_last6m_max_dpd",
        "bb_last6m_rate",
        "bb_last12m_max_dpd",
        "bb_last12m_rate",
        "bb_months_balance_max",
        "bb_months_balance_min",
        "bb_dpd_trend",
        "bb_ever_dpd_ge2",
        "bb_ever_dpd_ge5",
        "bb_improving",
        "bb_slope_reversal",
        "bb_status_0_count",
        "bb_closed_early",
        "bb_first_half_mean_dpd",
        "bb_second_half_mean_dpd",
        "bb_consec_bad_max",
        "bb_consec_good_max",
        "bb_streak_end_dpd",
    ]


BureauBalanceAggregator = DefaultBureauBalanceAggregator
