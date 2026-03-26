"""Feature transformers for creating derived features."""

import polars as pl
from polars import DataFrame


class FeatureTransformer:
    def add_ratio_features(self, df: DataFrame) -> DataFrame:
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO")
            )
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO")
            )
        if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_CREDIT") / pl.col("AMT_ANNUITY")).alias("CREDIT_ANNUITY_RATIO")
            )
        return df

    def add_days_features(self, df: DataFrame) -> DataFrame:
        if "DAYS_BIRTH" in df.columns:
            df = df.with_columns((pl.col("DAYS_BIRTH") / -365).alias("YEARS_BIRTH"))
        if "DAYS_EMPLOYED" in df.columns:
            df = df.with_columns((pl.col("DAYS_EMPLOYED") / -365).alias("YEARS_EMPLOYED"))
        if "DAYS_ID_PUBLISH" in df.columns:
            df = df.with_columns((pl.col("DAYS_ID_PUBLISH") / -365).alias("YEARS_ID_PUBLISH"))
        if "YEARS_BIRTH" in df.columns and "YEARS_EMPLOYED" in df.columns:
            df = df.with_columns(
                (pl.col("YEARS_BIRTH") - pl.col("YEARS_EMPLOYED")).alias("EMPLOYED_BIRTH_RATIO")
            )
        return df

    def add_bureau_features(self, df: DataFrame) -> DataFrame:
        if "DAYS_CREDIT_mean" in df.columns:
            df = df.with_columns(
                (pl.col("DAYS_CREDIT_mean") / -365).alias("YEARS_SINCE_CREDIT_mean")
            )
        if "AMT_CREDIT_SUM_mean" in df.columns and "AMT_CREDIT_SUM_sum" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_CREDIT_SUM_mean") / pl.col("AMT_CREDIT_SUM_sum")).alias(
                    "CREDIT_SUM_RATIO"
                )
            )
        return df

    def add_previous_app_features(self, df: DataFrame) -> DataFrame:
        if "AMT_APPLICATION_mean" in df.columns and "AMT_CREDIT_mean" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_APPLICATION_mean") / pl.col("AMT_CREDIT_mean")).alias(
                    "APPLICATION_CREDIT_RATIO"
                )
            )
        if "AMT_DOWN_PAYMENT_mean" in df.columns and "AMT_CREDIT_mean" in df.columns:
            df = df.with_columns(
                (pl.col("AMT_DOWN_PAYMENT_mean") / pl.col("AMT_CREDIT_mean")).alias(
                    "DOWN_PAYMENT_RATIO"
                )
            )
        return df

    def add_installments_features(self, df: DataFrame) -> DataFrame:
        if "PAYMENT_DIFF_mean" in df.columns and "AMT_PAYMENT_sum" in df.columns:
            df = df.with_columns(
                (pl.col("PAYMENT_DIFF_mean") / pl.col("AMT_PAYMENT_sum")).alias(
                    "PAYMENT_DIFF_RATIO"
                )
            )
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        df = self.add_ratio_features(df)
        df = self.add_days_features(df)
        if any("bureau" in col for col in df.columns):
            df = self.add_bureau_features(df)
        if any("previous" in col for col in df.columns):
            df = self.add_previous_app_features(df)
        if "PAYMENT_DIFF_mean" in df.columns:
            df = self.add_installments_features(df)
        return df
