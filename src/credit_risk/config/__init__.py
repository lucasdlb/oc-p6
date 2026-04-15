"""Configuration package.

Exports:
  - cfg: Global Config instance
  - Config: Pydantic model for config validation
  - load_config(): Function to load/reload config
  - ConfigGrid: For hyperparameter sweeps
"""

from __future__ import annotations

from credit_risk.config.config import cfg, load_config, reload_config
from credit_risk.config.config_grid import ConfigGrid
from credit_risk.config.models import (
    AggregationConfig,
    Config,
    DataConfig,
    FeaturesConfig,
    ImportanceConfig,
    InterpretConfig,
    MLFlowConfig,
    ModelConfig,
    RunConfig,
    SearchConfig,
    SelectionConfig,
    SplitterConfig,
    TuningConfig,
)

__all__ = [
    # Global config
    "cfg",
    "load_config",
    "reload_config",
    # Config models
    "Config",
    "RunConfig",
    "SplitterConfig",
    "ModelConfig",
    "SelectionConfig",
    "ImportanceConfig",
    "MLFlowConfig",
    "InterpretConfig",
    "SearchConfig",
    "TuningConfig",
    "DataConfig",
    "FeaturesConfig",
    "AggregationConfig",
    # Hyperparameter sweeps
    "ConfigGrid",
]
