"""Tests for data loading and processing."""

import polars as pl

from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoder import CategoricalEncoder


def test_cleaner_handles_anomalous_days_employed():
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_EMPLOYED": [365243, -1000, -2000],
        }
    )
    cleaner = DataCleaner()
    result = cleaner.clean_application(df)
    assert result["DAYS_EMPLOYED"].null_count() == 1
    assert result.filter(pl.col("DAYS_EMPLOYED").is_not_null())["DAYS_EMPLOYED"][0] == -1000


def test_encoder_categorical_columns(sample_dataframe):
    df = sample_dataframe.with_columns(pl.Series("CATEGORY", ["A", "B", "A"]))
    encoder = CategoricalEncoder()
    cols = encoder.get_categorical_columns(df)
    assert "CATEGORY" in cols


def test_encoder_low_cardinality_columns(sample_dataframe):
    df = sample_dataframe.with_columns(pl.Series("CATEGORY", ["A", "B", "A"]))
    encoder = CategoricalEncoder()
    cols = encoder.get_low_cardinality_columns(df)
    assert "CATEGORY" in cols
