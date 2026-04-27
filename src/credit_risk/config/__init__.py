"""Configuration package.

Usage:
    # Inference only - no optional steps
    from credit_risk.config import load_config
    cfg = load_config()

    # Training with optional configs - just declare which ones you need
    cfg = load_config("tuning", "selection")

    # Full pipeline
    cfg = load_config("tuning", "selection", "resampling", "interpret", "importance")
"""

from __future__ import annotations

from credit_risk.config.config import load_config
from credit_risk.config.config_grid import ConfigGrid
from credit_risk.config.models import (
    Config,
    DataConfig,
    DataSourceFiles,
    FeaturesConfig,
    ImportanceConfig,
    InterpretConfig,
    ModelConfig,
    OutputConfig,
    ResamplingConfig,
    RunConfig,
    SelectionConfig,
    SplitterConfig,
    TableConfig,
    TargetConfig,
    TuningConfig,
)

__all__ = [
    "load_config",
    "ConfigGrid",
    "Config",
    "RunConfig",
    "SplitterConfig",
    "ModelConfig",
    "SelectionConfig",
    "ImportanceConfig",
    "InterpretConfig",
    "TuningConfig",
    "DataConfig",
    "DataSourceFiles",
    "FeaturesConfig",
    "TableConfig",
    "TargetConfig",
    "ResamplingConfig",
    "OutputConfig",
]
