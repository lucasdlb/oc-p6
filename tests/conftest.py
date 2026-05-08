"""Pytest configuration and fixtures."""

import os

import pytest

# Use the test data config for all tests so that column-presence tests
# always exercise real transformers regardless of what data.toml says.
# Individual tests that need a NoOpStep can override via monkeypatch.
os.environ.setdefault("DATA_CONFIG", "data/test")


@pytest.fixture
def sample_config():
    from credit_risk.config import Config, ModelConfig

    return Config(
        model=ModelConfig(n_estimators=10),
    )


@pytest.fixture
def sample_dataframe():
    import polars as pl

    return pl.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "AMT_CREDIT": [100000.0, 200000.0, 150000.0],
            "AMT_INCOME_TOTAL": [50000.0, 80000.0, 60000.0],
            "AMT_ANNUITY": [5000.0, 10000.0, 7500.0],
            "DAYS_BIRTH": [-10000, -15000, -12000],
            "DAYS_EMPLOYED": [-2000, -3000, -1500],
        }
    )
