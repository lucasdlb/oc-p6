"""Transformer for previous_application table features."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class PreviousApplicationTransformer(StatelessStep):
    """Transformer for previous_application aggregated features.

    Operates on SK_ID_CURR-level aggregates produced by PreviousApplicationAggregator.
    All input columns are already aggregated (mean, sum, etc.) — no raw rows here.
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        def has(*names: str) -> bool:
            return all(c in X.columns for c in names)

        new_cols = []

        # --- credit fulfilment: how much of what was requested was granted ---
        if has("prev_AMT_CREDIT_mean", "prev_AMT_APPLICATION_mean"):
            new_cols.append(
                (pl.col("prev_AMT_CREDIT_mean") / (pl.col("prev_AMT_APPLICATION_mean") + 1)).alias(
                    "prev_credit_fulfilment_rate"
                )
                # < 1: consistently granted less than requested — lender caution signal
                # > 1: granted more than requested — upsell, lower risk signal
            )

        # --- annuity burden on previous loans ---
        if has("prev_AMT_ANNUITY_mean", "prev_AMT_CREDIT_mean"):
            new_cols.append(
                (pl.col("prev_AMT_ANNUITY_mean") / (pl.col("prev_AMT_CREDIT_mean") + 1)).alias(
                    "prev_annuity_to_credit_ratio"
                )
            )

        # --- down payment commitment rate ---
        if has("prev_AMT_DOWN_PAYMENT_mean", "prev_AMT_APPLICATION_mean"):
            new_cols.append(
                (
                    pl.col("prev_AMT_DOWN_PAYMENT_mean") / (pl.col("prev_AMT_APPLICATION_mean") + 1)
                ).alias("prev_down_payment_rate")
            )

        # --- refusal pressure: refused amount relative to approved ---
        if has("prev_refused_amt_sum", "prev_AMT_CREDIT_sum"):
            new_cols.append(
                (pl.col("prev_refused_amt_sum") / (pl.col("prev_AMT_CREDIT_sum") + 1)).alias(
                    "prev_refused_to_approved_amt_ratio"
                )
                # high ratio = a lot of credit was refused relative to what was granted
            )

        # --- application frequency: how actively was credit sought ---
        if has("prev_n_records", "prev_DAYS_DECISION_min", "prev_DAYS_DECISION_max"):
            credit_history_span = (
                pl.col("prev_DAYS_DECISION_max") - pl.col("prev_DAYS_DECISION_min")
            ).abs() + 1
            new_cols.append(
                (pl.col("prev_n_records") / credit_history_span * 365).alias(
                    "prev_applications_per_year"
                )
                # high = serial applicant — can signal financial stress
            )

        # --- recent refusal pressure ---
        if has("prev_refused_count_1y", "prev_n_applications_1y"):
            new_cols.append(
                (pl.col("prev_refused_count_1y") / (pl.col("prev_n_applications_1y") + 1)).alias(
                    "prev_refusal_rate_1y"
                )
            )

        # --- approved credit trend: recent vs historical ---
        if has("prev_amt_credit_mean_1y", "prev_AMT_CREDIT_mean"):
            new_cols.append(
                (pl.col("prev_amt_credit_mean_1y") / (pl.col("prev_AMT_CREDIT_mean") + 1)).alias(
                    "prev_credit_recency_ratio"
                )
                # > 1: applying for more recently — increasing financial need signal
            )

        if new_cols:
            X = X.with_columns(new_cols)

        return X
