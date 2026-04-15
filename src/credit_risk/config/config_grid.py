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
    """Return a copy of d with the dotted key set to value."""
    d = copy.deepcopy(d)
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return d


class ConfigGrid:
    """Parses a TOML file where list values define grid axes.

    Iterates over the cartesian product as validated Config instances.

    Usage:
        grid = ConfigGrid("configs/debug.toml")
        print(f"Running {len(grid)} experiments")

        for i, cfg in enumerate(grid):
            print(f"[{i+1}/{len(grid)}] method={cfg.resampling.method}, k={cfg.resampling.k_neighbors}")
            run_experiment(cfg)
    """

    def __init__(self, path: str | Path):
        with open(path, "rb") as f:
            self._raw = tomllib.load(f)
        self._axes = _find_axes(self._raw)

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
        if not self._axes:
            yield Config(**self._raw)
            return

        keys = list(self._axes.keys())
        values = list(self._axes.values())

        for combo in itertools.product(*values):
            raw = self._raw
            for k, v in zip(keys, combo):
                raw = _set_nested(raw, k, v)
            yield Config.model_validate(raw)

    def __len__(self) -> int:
        return self.n_configs
