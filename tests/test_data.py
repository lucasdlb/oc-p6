"""Tests for data loading and processing."""

import polars as pl
import pytest

from credit_risk.data.cleaning.registry import CleaningRegistry
from credit_risk.data.encoding import CategoricalEncoder


def test_cleaner_handles_anomalous_days_employed():
    df = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_EMPLOYED": [365243, -1000, -2000],
        }
    )
    cleaner = CleaningRegistry.get("ApplicationCleaner")()
    result = cleaner.fit_transform(df)
    assert result["is_never_employed"].null_count() == 0
    assert result["is_never_employed"].to_list() == [1, 0, 0]
    assert result["YEARS_EMPLOYED"].null_count() == 1
    assert result.filter(pl.col("YEARS_EMPLOYED").is_not_null())["YEARS_EMPLOYED"][
        0
    ] == pytest.approx(2.74, rel=1e-2)


def test_encoder_fit_transform(sample_dataframe):
    df = sample_dataframe.with_columns(pl.Series("CATEGORY", ["A", "B", "A"]))
    encoder = CategoricalEncoder()
    result = encoder.fit_transform(df)
    assert "CATEGORY" in result.columns
