"""Configuration package.

Exports:
  - cfg: Global Config instance
  - Config: Pydantic model for config validation
  - load_config(): Function to load/reload config
  - ModelGridConfig, REGISTRY, PRESETS, get_configs(), trim_grid(): Grid search registry
"""

from __future__ import annotations

from credit_risk.config.config import cfg, load_config, reload_config
from credit_risk.config.models import (
    AggregationConfig,
    Config,
    DataConfig,
    DataSourcesConfig,
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
from credit_risk.config.registry import (
    PRESETS,
    REGISTRY,
    ModelGridConfig,
    get_all_configs,
    get_configs,
    trim_grid,
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
    "DataSourcesConfig",
    # Registry
    "REGISTRY",
    "PRESETS",
    "ModelGridConfig",
    "get_configs",
    "get_all_configs",
    "trim_grid",
]
