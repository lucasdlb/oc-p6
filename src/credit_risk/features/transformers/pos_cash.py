"""Transformer for POS CASH table features."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk.features.transformers.base import TableTransformer


class POSCashTransformer(TableTransformer):
    """Transformer for POS_CASH_balance table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform POS_CASH_balance features.

        Args:
            df: Input dataframe with POS_CASH_balance aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = df.columns
        new_cols = []

        if "pos_SK_DPD_mean" in cols and "pos_SK_DPD_DEF_mean" in cols:
            new_cols.append(
                (pl.col("pos_SK_DPD_mean") / (pl.col("pos_SK_DPD_DEF_mean") + 1)).alias(
                    "pos_dpd_to_def_ratio"
                )
            )

        if "pos_CNT_INSTALMENT_FUTURE_mean" in cols and "pos_CNT_INSTALMENT_mean" in cols:
            new_cols.append(
                (
                    pl.col("pos_CNT_INSTALMENT_FUTURE_mean")
                    / (pl.col("pos_CNT_INSTALMENT_mean") + 1)
                ).alias("pos_future_to_total_instalment_ratio")
            )

        if "pos_MONTHS_BALANCE_max" in cols and "pos_MONTHS_BALANCE_min" in cols:
            new_cols.append(
                (pl.col("pos_MONTHS_BALANCE_max") - pl.col("pos_MONTHS_BALANCE_min")).alias(
                    "pos_history_span"
                )
            )

        if "pos_SK_DPD_mean" in cols and "pos_n_records" in cols:
            new_cols.append(
                (pl.col("pos_SK_DPD_mean") * pl.col("pos_n_records")).alias("pos_dpd_intensity")
            )

        if new_cols:
            df = df.with_columns(new_cols)

        return df
