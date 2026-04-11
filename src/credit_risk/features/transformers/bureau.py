"""Transformer for bureau table features."""

from __future__ import annotations

import polars as pl
from polars import DataFrame

from credit_risk.features.transformers.base import TableTransformer


class BureauTransformer(TableTransformer):
    """Transformer for bureau table features.

    Applies feature engineering transformations:
    - Feature interactions
    - Ratios
    - Derived features
    """

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform bureau features.

        Args:
            df: Input dataframe with bureau aggregated features

        Returns:
            Transformed dataframe with new engineered features
        """
        cols = df.columns
        new_cols = []

        if "bureau_DAYS_CREDIT_mean" in cols and "bureau_DAYS_CREDIT_max" in cols:
            new_cols.append(
                (pl.col("bureau_DAYS_CREDIT_mean") / (pl.col("bureau_DAYS_CREDIT_max") + 1)).alias(
                    "bureau_avg_to_max_credit_age"
                )
            )

        if "bureau_AMT_CREDIT_SUM_mean" in cols and "bureau_AMT_ANNUITY_mean" in cols:
            new_cols.append(
                (
                    pl.col("bureau_AMT_CREDIT_SUM_mean") / (pl.col("bureau_AMT_ANNUITY_mean") + 1)
                ).alias("bureau_credit_to_annuity_ratio")
            )

        if "bureau_AMT_CREDIT_SUM_OVERDUE_mean" in cols and "bureau_AMT_CREDIT_SUM_mean" in cols:
            new_cols.append(
                (
                    pl.col("bureau_AMT_CREDIT_SUM_OVERDUE_mean")
                    / (pl.col("bureau_AMT_CREDIT_SUM_mean") + 1)
                ).alias("bureau_overdue_ratio")
            )

        if "bureau_CREDIT_DAY_OVERDUE_max" in cols and "bureau_CREDIT_DAY_OVERDUE_mean" in cols:
            new_cols.append(
                (
                    pl.col("bureau_CREDIT_DAY_OVERDUE_max")
                    * pl.col("bureau_CREDIT_DAY_OVERDUE_mean")
                ).alias("bureau_overdue_severity")
            )

        if "bureau_CNT_CREDIT_PROLONG_sum" in cols and "bureau_n_records" in cols:
            new_cols.append(
                (pl.col("bureau_CNT_CREDIT_PROLONG_sum") / (pl.col("bureau_n_records") + 1)).alias(
                    "bureau_avg_prolong_per_credit"
                )
            )

        if new_cols:
            df = df.with_columns(new_cols)

        return df
