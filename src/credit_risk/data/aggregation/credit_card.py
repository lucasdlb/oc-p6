"""Credit card balance table aggregator — competition-grade feature engineering.

Three aggregation levels:
    MinimalCreditCardAggregator   : 18 features — fast iteration
    DefaultCreditCardAggregator   : 32 features — standard training
    DetailedCreditCardAggregator   : 44 features — full feature search

Feature groups
--------------
Status     : contract status counts and rates (Active, Completed, Demand, Signed...)
Utilization: balance-to-limit ratio, overlimit events, trend
Payment    : minimum payment ratio, payment-to-balance behavior
DPD        : days-past-due statistics and recency
Receivable : principal vs total receivable ratio
"""

from __future__ import annotations

import polars as pl
from polars import DataFrame, LazyFrame

from credit_risk.data.aggregation._common import (
    _categorical_diversity_exprs,
    _get_agg_exprs,
    _recency_weighted_mean,
)
from credit_risk.data.base import StatelessStep


class _BaseCreditCardAggregator(StatelessStep):
    """Base class for credit card aggregators.

    Daughter classes set FEATURES (list of feature names to compute).
    """

    FEATURES: list[str] = []

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        lf = X.lazy()
        lf = self._add_derived_columns(lf)

        schema = dict(X.schema)
        agg_cols = [c for c in X.columns if c not in {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}]

        exprs = (
            _get_agg_exprs(agg_cols, schema, "cc_", self._agg_methods())
            + self._status_exprs()
            + self._utilization_exprs()
            + self._payment_exprs()
            + self._dpd_exprs()
            + self._receivable_exprs()
        )
        exprs.append(pl.len().alias("cc_n_records"))

        result = lf.group_by("SK_ID_CURR").agg(*exprs)
        result = result.join(self._categorical_diversity(lf), on="SK_ID_CURR", how="left")

        return result.collect()

    def _add_derived_columns(self, lf: LazyFrame) -> LazyFrame:
        return lf.with_columns(
            (pl.col("AMT_BALANCE") / pl.col("AMT_CREDIT_LIMIT_ACTUAL").clip(lower_bound=1)).alias(
                "cc_utilization"
            ),
            (
                pl.col("AMT_PAYMENT_CURRENT")
                / pl.col("AMT_INST_MIN_REGULARITY").clip(lower_bound=0.01)
            ).alias("cc_min_payment_ratio"),
            pl.col("AMT_BALANCE").alias("cc_balance"),
            pl.col("AMT_CREDIT_LIMIT_ACTUAL").alias("cc_limit"),
        )

    def _agg_methods(self) -> list[str]:
        from credit_risk.data.aggregation._common import AGG_METHODS

        return list(AGG_METHODS["default"])

    def _status_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "cc_completed_count" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Completed").sum().alias("cc_completed_count")
            )
        if "cc_active_count" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("cc_active_count")
            )
        if "cc_demand_count" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Demand").sum().alias("cc_demand_count")
            )
        if "cc_signed_count" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Signed").sum().alias("cc_signed_count")
            )
        if "cc_proposal_count" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Sent proposal").sum().alias("cc_proposal_count")
            )
        if "cc_completion_rate" in self.FEATURES:
            exprs.append(
                ((pl.col("NAME_CONTRACT_STATUS") == "Completed").sum() / (pl.len() + 1)).alias(
                    "cc_completion_rate"
                )
            )
        if "cc_ever_demand" in self.FEATURES:
            exprs.append(
                (pl.col("NAME_CONTRACT_STATUS") == "Demand")
                .any()
                .cast(pl.Int8)
                .alias("cc_ever_demand")
            )
        return exprs

    def _utilization_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "cc_utilization_mean" in self.FEATURES:
            exprs.append(pl.col("cc_utilization").mean().alias("cc_utilization_mean"))
        if "cc_utilization_max" in self.FEATURES:
            exprs.append(pl.col("cc_utilization").max().alias("cc_utilization_max"))
        if "cc_overlimit_count" in self.FEATURES:
            exprs.append((pl.col("cc_utilization") > 1.0).sum().alias("cc_overlimit_count"))
        if "cc_utilization_trend" in self.FEATURES:
            exprs.append(
                pl.cov(pl.col("MONTHS_BALANCE"), pl.col("cc_utilization")).alias(
                    "cc_utilization_trend"
                )
            )
        if "cc_last6m_utilization" in self.FEATURES:
            exprs.append(
                pl.col("cc_utilization")
                .filter(pl.col("MONTHS_BALANCE") >= -6)
                .mean()
                .alias("cc_last6m_utilization")
            )
        if "cc_utilization_std" in self.FEATURES:
            exprs.append(pl.col("cc_utilization").std().alias("cc_utilization_std"))
        if "cc_utilization_recent" in self.FEATURES:
            exprs.append(
                _recency_weighted_mean("cc_utilization", "MONTHS_BALANCE").alias(
                    "cc_utilization_recent"
                )
            )
        return exprs

    def _payment_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "cc_min_payment_ratio_mean" in self.FEATURES:
            exprs.append(pl.col("cc_min_payment_ratio").mean().alias("cc_min_payment_ratio_mean"))
        if "cc_min_payment_only_count" in self.FEATURES:
            exprs.append(
                (pl.col("cc_min_payment_ratio") < 1.0)
                .filter(pl.col("cc_min_payment_ratio").is_not_null())
                .sum()
                .alias("cc_min_payment_only_count")
            )
        if "cc_payment_to_balance_mean" in self.FEATURES:
            exprs.append(
                (pl.col("AMT_PAYMENT_CURRENT") / pl.col("cc_balance").clip(lower_bound=0.01))
                .mean()
                .alias("cc_payment_to_balance_mean")
            )
        return exprs

    def _dpd_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "cc_dpd_max" in self.FEATURES:
            exprs.append(pl.col("SK_DPD").max().alias("cc_dpd_max"))
        if "cc_dpd_mean" in self.FEATURES:
            exprs.append(pl.col("SK_DPD").mean().alias("cc_dpd_mean"))
        if "cc_dpd_months_count" in self.FEATURES:
            exprs.append((pl.col("SK_DPD") > 0).sum().alias("cc_dpd_months_count"))
        if "cc_dpd_def_max" in self.FEATURES:
            exprs.append(pl.col("SK_DPD_DEF").max().alias("cc_dpd_def_max"))
        if "cc_ever_overdue" in self.FEATURES:
            exprs.append((pl.col("SK_DPD") > 0).any().cast(pl.Int8).alias("cc_ever_overdue"))
        return exprs

    def _receivable_exprs(self) -> list[pl.Expr]:
        exprs = []
        if "cc_principal_ratio" in self.FEATURES:
            exprs.append(
                (
                    pl.col("AMT_RECEIVABLE_PRINCIPAL")
                    / pl.col("AMT_TOTAL_RECEIVABLE").clip(lower_bound=0.01)
                )
                .mean()
                .alias("cc_principal_ratio")
            )
        return exprs

    def _categorical_diversity(self, lf: LazyFrame) -> LazyFrame:
        return _categorical_diversity_exprs(lf, "SK_ID_CURR", "NAME_CONTRACT_STATUS", "cc_")

    def _load_config(self):
        from credit_risk.config import load_config

        return load_config().data.features


class MinimalCreditCardAggregator(_BaseCreditCardAggregator):
    """Minimal credit card aggregator — 18 features for fast iteration."""

    FEATURES = [
        "cc_completed_count",
        "cc_active_count",
        "cc_completion_rate",
        "cc_utilization_mean",
        "cc_utilization_max",
        "cc_dpd_max",
        "cc_dpd_mean",
        "cc_dpd_months_count",
        "cc_dpd_def_max",
        "cc_ever_overdue",
        "cc_payment_to_balance_mean",
        "cc_NAME_CONTRACT_STATUS_n_unique",
        "cc_NAME_CONTRACT_STATUS_mode",
    ]


class DefaultCreditCardAggregator(_BaseCreditCardAggregator):
    """Default credit card aggregator — 32 features for standard training."""

    FEATURES = [
        "cc_completed_count",
        "cc_active_count",
        "cc_demand_count",
        "cc_signed_count",
        "cc_proposal_count",
        "cc_completion_rate",
        "cc_ever_demand",
        "cc_utilization_mean",
        "cc_utilization_max",
        "cc_overlimit_count",
        "cc_utilization_trend",
        "cc_utilization_recent",
        "cc_last6m_utilization",
        "cc_utilization_std",
        "cc_min_payment_ratio_mean",
        "cc_min_payment_only_count",
        "cc_payment_to_balance_mean",
        "cc_dpd_max",
        "cc_dpd_mean",
        "cc_dpd_months_count",
        "cc_dpd_def_max",
        "cc_ever_overdue",
        "cc_principal_ratio",
        "cc_NAME_CONTRACT_STATUS_n_unique",
        "cc_NAME_CONTRACT_STATUS_mode",
    ]


class DetailedCreditCardAggregator(_BaseCreditCardAggregator):
    """Detailed credit card aggregator — 44 features for full feature search."""

    FEATURES = [
        "cc_completed_count",
        "cc_active_count",
        "cc_demand_count",
        "cc_signed_count",
        "cc_proposal_count",
        "cc_completion_rate",
        "cc_ever_demand",
        "cc_utilization_mean",
        "cc_utilization_max",
        "cc_overlimit_count",
        "cc_utilization_trend",
        "cc_utilization_recent",
        "cc_last6m_utilization",
        "cc_utilization_std",
        "cc_min_payment_ratio_mean",
        "cc_min_payment_only_count",
        "cc_payment_to_balance_mean",
        "cc_dpd_max",
        "cc_dpd_mean",
        "cc_dpd_months_count",
        "cc_dpd_def_max",
        "cc_ever_overdue",
        "cc_principal_ratio",
        "cc_NAME_CONTRACT_STATUS_n_unique",
        "cc_NAME_CONTRACT_STATUS_mode",
    ]


CreditCardBalanceAggregator = DefaultCreditCardAggregator
