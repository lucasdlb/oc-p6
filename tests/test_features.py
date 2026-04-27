"""Tests for feature engineering."""

import polars as pl

from credit_risk.data.transformation.registry import TransformerRegistry


def test_transformer_transforms_application_table(sample_dataframe):
    df = sample_dataframe.with_columns(
        pl.col("DAYS_BIRTH").cast(pl.Float64),
        pl.col("DAYS_EMPLOYED").cast(pl.Float64),
    )
    transformer = TransformerRegistry.get("ApplicationTransformer")()
    result = transformer.fit_transform(df)
    assert result is not None
    assert len(result) == len(df)
