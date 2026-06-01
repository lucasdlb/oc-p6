"""Config grid for hyperparameter sweeps."""

from __future__ import annotations

import copy
import itertools
import tomllib
from pathlib import Path
from typing import Any, Generator

from credit_risk.config.models import Config  # isort: skip

# Fields typed as list in pydantic configs - skip these in axis detection
LIST_FIELDS = {"tuning.models"}


def _find_axes(d: dict, prefix: str = "") -> dict[str, list]:
    """Find keys with list values that should become grid axes.

    Recurses into nested dicts.
    """
    axes = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if full_key in LIST_FIELDS:
            continue

        if isinstance(v, dict):
            # Recurse into nested dict
            axes.update(_find_axes(v, full_key))
            continue

        if not isinstance(v, list):
            continue
        if not v:
            continue

        # Found a list - check first element
        first = v[0]
        if isinstance(first, str):
            # List of strings - treat as axis if more than one element
            if len(v) > 1:
                axes[full_key] = v
        elif isinstance(first, (int, float, bool)):
            # Numeric list - treat as axis
            axes[full_key] = v
    return axes


def _set_nested(d: dict, dotted_key: str, value: Any) -> dict:
    """Return a copy of d with the dotted key set to value, creating nested dicts as needed."""
    d = copy.deepcopy(d)
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    return d


def _ensure_scalar(d: dict) -> dict:
    """Convert single-element lists to scalars for non-axis fields."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _ensure_scalar(v)
        elif isinstance(v, list) and len(v) == 1:
            result[k] = v[0]
        else:
            result[k] = v
    return result


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class ConfigGrid:
    """Parses a TOML file where list values define grid axes.

    Iterates over the cartesian product as validated Config instances.
    Optionally merges with a base config instead of creating standalone configs.

    Usage:
        grid = ConfigGrid("configs/sweep.toml", base_config=global_cfg)
        print(f"Running {len(grid)} experiments")

        for i, cfg in enumerate(grid):
            print(f"[{i+1}/{len(grid)}] method={cfg.cleaner.method}")
            run_experiment(cfg)
    """

    def __init__(self, path: str | Path, base_config: Config | None = None):
        with open(path, "rb") as f:
            self._raw = tomllib.load(f)
        self._axes = _find_axes(self._raw)
        self._base_config = base_config

    @property
    def n_configs(self) -> int:
        if not self._axes:
            return 1
        result = 1
        for values in self._axes.values():
            result *= len(values)
        return result

    @property
    def axes(self) -> dict[str, list]:
        """The grid axes: {dotted_key: [values]}."""
        return self._axes

    def __iter__(self) -> Generator[Config, None, None]:
        # Start with base config if provided, otherwise use model construct
        if self._base_config is not None:
            base_dict = self._base_config.model_dump()
        else:
            base_dict = {}

        if not self._axes:
            # Merge base with sweep config
            merged = _merge_dicts(base_dict, self._raw)
            cfg = Config.model_validate(_ensure_scalar(merged))
            yield cfg
            return

        keys = list(self._axes.keys())
        values = list(self._axes.values())

        for combo in itertools.product(*values):
            # Start with base config
            merged = copy.deepcopy(base_dict)
            # Apply grid values (override)
            for k, v in zip(keys, combo, strict=True):
                merged = _set_nested(merged, k, v)
            cfg = Config.model_validate(_ensure_scalar(merged))
            yield cfg

    def __len__(self) -> int:
        return self.n_configs
