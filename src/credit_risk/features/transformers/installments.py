"""Transformer for installments_payments table features."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk.features.transformers.base import TableTransformer


class InstallmentsTransformer(TableTransformer):
    """Transformer for installments_payments table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform installments_payments features.

        Args:
            df: Input dataframe with installments_payments aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = df.columns
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

        if new_cols:
            df = df.with_columns(new_cols)

        return df
