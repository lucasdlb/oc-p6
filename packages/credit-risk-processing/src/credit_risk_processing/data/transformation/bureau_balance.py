"""Transformer for bureau_balance table features."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk_processing.data.base import StatelessStep


class BureauBalanceTransformer(StatelessStep):
    """Transformer for bureau_balance table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """Transform bureau_balance features.

        Args:
            X: Input dataframe with bureau_balance aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = X.columns
        new_cols = []

        if "bb_status_0_count" in cols and "bb_status_C_count" in cols:
            new_cols.append(
                (pl.col("bb_status_0_count") / (pl.col("bb_status_C_count") + 1)).alias(
                    "bb_status_0_to_C_ratio"
                )
            )

        if "bb_mean_dpd" in cols and "bb_max_dpd" in cols:
            new_cols.append((pl.col("bb_mean_dpd") * pl.col("bb_max_dpd")).alias("bb_dpd_severity"))

        if "bb_delinquency_rate" in cols and "bb_credit_history_len" in cols:
            new_cols.append(
                (pl.col("bb_delinquency_rate") * pl.col("bb_credit_history_len")).alias(
                    "bb_delinquency_intensity"
                )
            )

        if "bb_avg_duration" in cols and "bb_n_credits" in cols:
            new_cols.append(
                (pl.col("bb_avg_duration") / (pl.col("bb_n_credits") + 1)).alias(
                    "bb_avg_duration_per_credit"
                )
            )

        if "bb_last12m_rate" in cols and "bb_credit_history_len" in cols:
            new_cols.append(
                (pl.col("bb_last12m_rate") * pl.col("bb_credit_history_len")).alias(
                    "bb_recent_activity_intensity"
                )
            )

        if "bb_pct_loans_ever_bad" in cols and "bb_n_credits" in cols:
            new_cols.append(
                (pl.col("bb_pct_loans_ever_bad") * pl.col("bb_n_credits")).alias(
                    "bb_bad_loan_concentration"
                )
            )

        if "bb_improving" in cols and "bb_dpd_trend" in cols:
            new_cols.append(
                (pl.col("bb_improving") + pl.col("bb_dpd_trend")).alias("bb_improvement_score")
            )

        if new_cols:
            X = X.with_columns(new_cols)

        return X
