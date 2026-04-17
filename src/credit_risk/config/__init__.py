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
    AggregationConfig,
    CleanerConfig,
    Config,
    DataConfig,
    FeaturesConfig,
    ImportanceConfig,
    ImputerConfig,
    InterpretConfig,
    ModelConfig,
    ResamplingConfig,
    RunConfig,
    SelectionConfig,
    SplitterConfig,
    TransformerConfig,
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
    "FeaturesConfig",
    "AggregationConfig",
    "ResamplingConfig",
    # Processing step configs
    "CleanerConfig",
    "ImputerConfig",
    "AggregatorConfig",
    "TransformerConfig",
]
