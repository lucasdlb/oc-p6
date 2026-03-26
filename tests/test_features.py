"""Tests for feature engineering."""

import polars as pl

from credit_risk.features.transformer import FeatureTransformer


def test_transformer_adds_ratio_features(sample_dataframe):
    transformer = FeatureTransformer()
    result = transformer.add_ratio_features(sample_dataframe)
    assert "CREDIT_INCOME_RATIO" in result.columns
    assert "ANNUITY_INCOME_RATIO" in result.columns
    assert "CREDIT_ANNUITY_RATIO" in result.columns


def test_transformer_adds_days_features(sample_dataframe):
    df = sample_dataframe.with_columns(
        [
            pl.col("DAYS_BIRTH").cast(pl.Float64),
            pl.col("DAYS_EMPLOYED").cast(pl.Float64),
        ]
    )
    transformer = FeatureTransformer()
    result = transformer.add_days_features(df)
    assert "YEARS_BIRTH" in result.columns
    assert "YEARS_EMPLOYED" in result.columns
    assert "EMPLOYED_BIRTH_RATIO" in result.columns
