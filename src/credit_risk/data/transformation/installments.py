"""Transformer for installments_payments table features."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class InstallmentsTransformer(StatelessStep):
    """Transformer for installments_payments table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """Transform installments_payments features.

        Args:
            X: Input dataframe with installments_payments aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = X.columns
        new_cols = []

        if "ins_AMT_PAYMENT_mean" in cols and "ins_AMT_INSTALMENT_mean" in cols:
            new_cols.append(
                (pl.col("ins_AMT_PAYMENT_mean") / (pl.col("ins_AMT_INSTALMENT_mean") + 1)).alias(
                    "ins_payment_to_instalment_ratio"
                )
            )

        if "ins_AMT_PAYMENT_mean" in cols and "ins_AMT_INSTALMENT_mean" in cols:
            new_cols.append(
                (pl.col("ins_AMT_PAYMENT_mean") - pl.col("ins_AMT_INSTALMENT_mean")).alias(
                    "ins_payment_diff"
                )
            )

        if "ins_NUM_INSTALMENT_NUMBER_mean" in cols and "ins_NUM_INSTALMENT_VERSION_mean" in cols:
            new_cols.append(
                (
                    pl.col("ins_NUM_INSTALMENT_NUMBER_mean")
                    * pl.col("ins_NUM_INSTALMENT_VERSION_mean")
                ).alias("ins_instalment_version_complexity")
            )

        if "ins_n_records" in cols:
            new_cols.append(
                pl.when(pl.col("ins_n_records") > 12)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .alias("ins_long_payment_history")
            )

        if "ins_late_payment_rate" in cols and "ins_underpayment_rate" in cols:
            new_cols.append(
                (pl.col("ins_late_payment_rate") * pl.col("ins_underpayment_rate")).alias(
                    "ins_late_underpay_interaction"
                )
            )

        if "ins_early_payment_rate" in cols and "ins_late_payment_rate" in cols:
            new_cols.append(
                (pl.col("ins_early_payment_rate") - pl.col("ins_late_payment_rate")).alias(
                    "ins_payment_behavior_score"
                )
            )

        if "ins_last12m_late_rate" in cols and "ins_late_payment_rate" in cols:
            new_cols.append(
                (pl.col("ins_last12m_late_rate") - pl.col("ins_late_payment_rate")).alias(
                    "ins_recent_vs_historical_late"
                )
            )

        if "ins_AMT_PAYMENT_sum" in cols and "ins_n_records" in cols:
            new_cols.append(
                (pl.col("ins_AMT_PAYMENT_sum") / (pl.col("ins_n_records") + 1)).alias(
                    "ins_avg_payment_per_record"
                )
            )

        if new_cols:
            X = X.with_columns(new_cols)

        return X
