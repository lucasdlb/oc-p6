"""Transformer for credit_card_balance table features."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep


class CreditCardBalanceTransformer(StatelessStep):
    """Transformer for credit_card_balance table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """Transform credit_card_balance features.

        Args:
            X: Input dataframe with credit_card_balance aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = X.columns
        new_cols = []

        if "cc_AMT_BALANCE_mean" in cols and "cc_AMT_CREDIT_LIMIT_ACTUAL_mean" in cols:
            new_cols.append(
                (
                    pl.col("cc_AMT_BALANCE_mean") / (pl.col("cc_AMT_CREDIT_LIMIT_ACTUAL_mean") + 1)
                ).alias("cc_balance_to_limit_ratio")
            )

        if "cc_AMT_DRAWINGS_CURRENT_mean" in cols and "cc_AMT_CREDIT_LIMIT_ACTUAL_mean" in cols:
            new_cols.append(
                (
                    pl.col("cc_AMT_DRAWINGS_CURRENT_mean")
                    / (pl.col("cc_AMT_CREDIT_LIMIT_ACTUAL_mean") + 1)
                ).alias("cc_utilization_ratio")
            )

        if "cc_AMT_PAYMENT_TOTAL_CURRENT_mean" in cols and "cc_AMT_PAYMENT_CURRENT_mean" in cols:
            new_cols.append(
                (
                    pl.col("cc_AMT_PAYMENT_TOTAL_CURRENT_mean")
                    / (pl.col("cc_AMT_PAYMENT_CURRENT_mean") + 1)
                ).alias("cc_total_to_current_payment_ratio")
            )

        if "cc_CNT_DRAWINGS_ATM_CURRENT_mean" in cols and "cc_CNT_DRAWINGS_CURRENT_mean" in cols:
            new_cols.append(
                (
                    pl.col("cc_CNT_DRAWINGS_ATM_CURRENT_mean")
                    / (pl.col("cc_CNT_DRAWINGS_CURRENT_mean") + 1)
                ).alias("cc_atm_to_total_drawings_ratio")
            )

        if "cc_SK_DPD_mean" in cols and "cc_SK_DPD_DEF_mean" in cols:
            new_cols.append(
                (pl.col("cc_SK_DPD_mean") / (pl.col("cc_SK_DPD_DEF_mean") + 1)).alias(
                    "cc_dpd_to_def_ratio"
                )
            )

        if "cc_SK_DPD_mean" in cols and "cc_n_records" in cols:
            new_cols.append(
                (pl.col("cc_SK_DPD_mean") * pl.col("cc_n_records")).alias("cc_dpd_intensity")
            )

        if new_cols:
            X = X.with_columns(new_cols)

        return X
