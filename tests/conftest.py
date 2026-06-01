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
