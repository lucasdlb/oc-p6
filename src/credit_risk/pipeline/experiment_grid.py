"""Experiment grid expansion engine."""

from __future__ import annotations

from itertools import product
from typing import Any

from credit_risk.config.experiment_config import (
    ExperimentConfig,
)


def _expand_value(value: Any) -> list[Any]:
    """Expand a single value or list into a list."""
    if isinstance(value, list):
        return value
    return [value]


def expand_data_sources(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Expand data sources from config into individual experiment configs.

    Handles lists in data_sources config to create grid search variants.
    """
    ds = config.data_sources
    grid_sources = [f for f in ds.model_fields if ds.is_grid(f)]

    if not grid_sources:
        return [config]

    grid_values = [ds.get_grid_values(src) for src in grid_sources]

    expanded = []
    for combo in product(*grid_values):
        new_ds = ds.model_copy(deep=True)
        name_parts = []

        for src, val in zip(grid_sources, combo, strict=True):
            new_ds = new_ds.to_single(src, val)
            name_parts.append(f"{src}_{val}")

        new_config = config.model_copy(deep=True)
        new_config.data_sources = new_ds
        new_config.name = f"{config.name}_{'_'.join(name_parts)}"

        expanded.append(new_config)

    return expanded


def expand_experiment_grid(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Expand an experiment config with search space into individual configs.

    Args:
        config: ExperimentConfig with search space

    Returns:
        List of concrete ExperimentConfigs to run
    """
    if not config.search:
        return expand_data_sources(config)

    search_keys = list(config.search.keys())
    search_values = [_expand_value(v) for v in config.search.values()]

    expanded = []
    for combo in product(*search_values):
        param_dict = dict(zip(search_keys, combo, strict=True))

        new_config = config.model_copy(deep=True)
        new_config.name = f"{config.name}_{'_'.join(str(v) for v in combo)}"

        for key, value in param_dict.items():
            if key.startswith("ds_"):
                source_name = key[3:]
                if hasattr(new_config.data_sources, source_name):
                    new_config.data_sources = new_config.data_sources.to_single(source_name, value)
            else:
                if key in new_config.model:
                    new_config.model[key] = value
                elif key in new_config.data:
                    new_config.data[key] = value
                else:
                    new_config.model[key] = value

        expanded.append(new_config)

    return expand_data_sources(new_config)


def generate_grid_search(
    base_config: ExperimentConfig,
    data_source_toggles: dict[str, list[bool]] | None = None,
    param_grid: dict[str, list[Any]] | None = None,
) -> list[ExperimentConfig]:
    """Generate grid search configurations.

    Args:
        base_config: Base experiment config
        data_source_toggles: Dict of {source: [True, False]} to grid over
        param_grid: Dict of {param: [values]} to grid over

    Returns:
        List of ExperimentConfigs for all combinations
    """
    if data_source_toggles is None and param_grid is None:
        return expand_data_sources(base_config)

    configs = [base_config]

    if data_source_toggles:
        new_configs = []
        for cfg in configs:
            for source, options in data_source_toggles.items():
                for option in options:
                    new_cfg = cfg.model_copy(deep=True)
                    new_cfg.name = f"{cfg.name}_{source}_{option}"
                    if hasattr(new_cfg.data_sources, source):
                        new_cfg.data_sources = new_cfg.data_sources.to_single(source, option)
                    new_configs.append(new_cfg)
        configs = new_configs

    if param_grid:
        final_configs = []
        for cfg in configs:
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            for combo in product(*param_values):
                new_cfg = cfg.model_copy(deep=True)
                new_cfg.name = f"{cfg.name}_{'_'.join(str(v) for v in combo)}"
                for key, value in zip(param_keys, combo, strict=True):
                    new_cfg.model[key] = value
                final_configs.append(new_cfg)
        return final_configs

    return expand_data_sources(configs[0]) if configs else []
