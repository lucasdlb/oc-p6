"""Installments payments table aggregator — competition-grade feature engineering.

Three aggregation levels (sweep/debug/competition):
    MinimalInstallmentsAggregator  : 5 features  — fast iteration / sanity checks
    DefaultInstallmentsAggregator  : 18 features — standard training runs
    DetailedInstallmentsAggregator  : 22 features — full feature search

Feature groups
--------------
Late payment  : fraction, severity, and recency of payments made after due date
Underpayment  : fraction and magnitude of payments below expected amount
Volatility    : payment amount stability and plan renegotiation frequency
Recency       : last-6m / last-12m windows weighted toward recent behaviour
Trajectory    : payment behaviour trend over time (improving vs deteriorating)
"""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk_processing.data.base import StatelessStep


class _BaseInstallmentsAggregator(StatelessStep):
    """Base class for installments aggregators.

    Daughter classes set FEATURES (list of feature names to compute).
    The base class handles the aggregation logic and returns one row per SK_ID_CURR.
    """

    FEATURES: list[str] = []

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        lf = X.lazy()
        lf = self._add_derived_columns(lf)
        exprs = (
            self._generic_agg_exprs(X)
            + self._late_payment_exprs()
            + self._underpayment_exprs()
            + self._volatility_exprs()
            + self._recency_exprs()
            + self._trajectory_exprs()
        )
        return lf.group_by("SK_ID_CURR").agg(*exprs, pl.len().alias("ins_n_records")).collect()

    def _add_derived_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("days_late"),
            (pl.col("AMT_PAYMENT") / pl.col("AMT_INSTALMENT").clip(lower_bound=0.01)).alias(
                "payment_ratio"
            ),
            pl.col("AMT_PAYMENT").alias("amt_payment"),
            pl.col("AMT_INSTALMENT").alias("amt_instalment"),
        )

    def _generic_agg_exprs(self, X: DataFrame) -> list[pl.Expr]:
        from credit_risk_processing.data.aggregation._common import _get_agg_exprs

        agg_cols = [c for c in X.columns if c not in {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}]
        agg_methods = ["mean", "sum", "min", "max", "std", "count"]
        return _get_agg_exprs(agg_cols, dict(X.schema), "ins_", agg_methods)

    def _late_payment_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "ins_late_payment_rate" in self.FEATURES:
            exprs.append((pl.col("days_late") > 0).mean().alias("ins_late_payment_rate"))
        if "ins_ever_late" in self.FEATURES:
            exprs.append((pl.col("days_late") > 0).any().cast(pl.Int8).alias("ins_ever_late"))
        if "ins_late_count" in self.FEATURES:
            exprs.append((pl.col("days_late") > 0).sum().alias("ins_late_count"))
        if "ins_avg_days_late" in self.FEATURES:
            exprs.append(
                pl.col("days_late")
                .filter(pl.col("days_late") > 0)
                .mean()
                .alias("ins_avg_days_late")
            )
        if "ins_max_days_late" in self.FEATURES:
            exprs.append(pl.col("days_late").max().alias("ins_max_days_late"))
        if "ins_late_payment_std" in self.FEATURES:
            exprs.append(
                pl.col("days_late")
                .filter(pl.col("days_late") > 0)
                .std()
                .alias("ins_late_payment_std")
            )
        return exprs

    def _underpayment_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "ins_underpayment_rate" in self.FEATURES:
            exprs.append(
                (pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT"))
                .mean()
                .alias("ins_underpayment_rate")
            )
        if "ins_overpayment_rate" in self.FEATURES:
            exprs.append(
                (pl.col("AMT_PAYMENT") > pl.col("AMT_INSTALMENT"))
                .mean()
                .alias("ins_overpayment_rate")
            )
        if "ins_avg_payment_ratio" in self.FEATURES:
            exprs.append(pl.col("payment_ratio").mean().alias("ins_avg_payment_ratio"))
        if "ins_min_payment_ratio" in self.FEATURES:
            exprs.append(pl.col("payment_ratio").min().alias("ins_min_payment_ratio"))
        if "ins_max_underpay_ratio" in self.FEATURES:
            exprs.append(
                (1 - pl.col("payment_ratio"))
                .filter(pl.col("payment_ratio") < 1)
                .max()
                .alias("ins_max_underpay_ratio")
            )
        if "ins_pct_zero_payments" in self.FEATURES:
            exprs.append((pl.col("AMT_PAYMENT") == 0).mean().alias("ins_pct_zero_payments"))
        return exprs

    def _volatility_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "ins_payment_std" in self.FEATURES:
            exprs.append(pl.col("amt_payment").std().alias("ins_payment_std"))
        if "ins_instalment_std" in self.FEATURES:
            exprs.append(pl.col("amt_instalment").std().alias("ins_instalment_std"))
        if "ins_payment_cv" in self.FEATURES:
            exprs.append(
                (pl.col("amt_payment").std() / (pl.col("amt_payment").mean() + 1)).alias(
                    "ins_payment_cv"
                )
            )
        if "ins_version_complexity" in self.FEATURES:
            exprs.append(
                pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("ins_version_complexity")
            )
        return exprs

    def _recency_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "ins_last12m_late_rate" in self.FEATURES:
            exprs.append(
                (pl.col("days_late") > 0)
                .filter(pl.col("DAYS_INSTALMENT") >= -365)
                .mean()
                .alias("ins_last12m_late_rate")
            )
        if "ins_last12m_payment_ratio" in self.FEATURES:
            exprs.append(
                pl.col("payment_ratio")
                .filter(pl.col("DAYS_INSTALMENT") >= -365)
                .mean()
                .alias("ins_last12m_payment_ratio")
            )
        if "ins_last6m_late_rate" in self.FEATURES:
            exprs.append(
                (pl.col("days_late") > 0)
                .filter(pl.col("DAYS_INSTALMENT") >= -180)
                .mean()
                .alias("ins_last6m_late_rate")
            )
        return exprs

    def _trajectory_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "ins_payment_trend" in self.FEATURES:
            exprs.append(
                pl.cov(pl.col("DAYS_INSTALMENT"), pl.col("payment_ratio")).alias(
                    "ins_payment_trend"
                )
            )
        if "ins_early_payment_rate" in self.FEATURES:
            exprs.append((pl.col("days_late") < 0).mean().alias("ins_early_payment_rate"))
        if "ins_avg_days_early" in self.FEATURES:
            exprs.append(
                (-pl.col("days_late"))
                .filter(pl.col("days_late") < 0)
                .mean()
                .alias("ins_avg_days_early")
            )
        return exprs




class MinimalInstallmentsAggregator(_BaseInstallmentsAggregator):
    """Minimal installments aggregator — 5 features for fast iteration."""

    FEATURES = [
        "ins_late_payment_rate",
        "ins_ever_late",
        "ins_underpayment_rate",
        "ins_payment_std",
    ]


class DefaultInstallmentsAggregator(_BaseInstallmentsAggregator):
    """Default installments aggregator — 18 features for standard training."""

    FEATURES = [
        "ins_late_payment_rate",
        "ins_ever_late",
        "ins_late_count",
        "ins_avg_days_late",
        "ins_max_days_late",
        "ins_underpayment_rate",
        "ins_overpayment_rate",
        "ins_avg_payment_ratio",
        "ins_min_payment_ratio",
        "ins_payment_std",
        "ins_instalment_std",
        "ins_payment_cv",
        "ins_version_complexity",
        "ins_last12m_late_rate",
        "ins_last12m_payment_ratio",
        "ins_last6m_late_rate",
        "ins_payment_trend",
        "ins_early_payment_rate",
    ]


class DetailedInstallmentsAggregator(_BaseInstallmentsAggregator):
    """Detailed installments aggregator — 22 features for full feature search."""

    FEATURES = [
        "ins_late_payment_rate",
        "ins_ever_late",
        "ins_late_count",
        "ins_avg_days_late",
        "ins_max_days_late",
        "ins_underpayment_rate",
        "ins_overpayment_rate",
        "ins_avg_payment_ratio",
        "ins_min_payment_ratio",
        "ins_payment_std",
        "ins_instalment_std",
        "ins_payment_cv",
        "ins_version_complexity",
        "ins_last12m_late_rate",
        "ins_last12m_payment_ratio",
        "ins_last6m_late_rate",
        "ins_payment_trend",
        "ins_early_payment_rate",
        "ins_late_payment_std",
        "ins_avg_days_early",
        "ins_max_underpay_ratio",
        "ins_pct_zero_payments",
    ]
