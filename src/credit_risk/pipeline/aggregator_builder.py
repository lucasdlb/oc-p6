"""Aggregator builder for creating pipeline configurations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    enabled: bool = True
    aggregation_method: str = "default"


@dataclass
class AggregatorBuilder:
    """Builder for creating aggregator/pipeline configurations.

    Usage:
        builder = AggregatorBuilder(bureau=[False, True], bureau_balance=True)
        for config in builder.build():
            pipeline = config.create_pipeline()
            result = pipeline.run()
    """

    application: bool | list[bool] = True
    bureau: bool | list[bool] = True
    bureau_balance: bool | list[bool] = True
    previous_application: bool | list[bool] = True
    pos_cash: bool | list[bool] = True
    installments: bool | list[bool] = True
    credit_card: bool | list[bool] = True

    def __post_init__(self):
        self._sources = [
            "application",
            "bureau",
            "bureau_balance",
            "previous_application",
            "pos_cash",
            "installments",
            "credit_card",
        ]

    def _normalize(self, value: bool | list[bool]) -> list[bool]:
        if isinstance(value, bool):
            return [value]
        return value

    def get_grid_values(self) -> dict[str, list[bool]]:
        return {src: self._normalize(getattr(self, src)) for src in self._sources}

    def has_grid(self) -> bool:
        return any(
            isinstance(getattr(self, src), list) and len(getattr(self, src)) > 1
            for src in self._sources
        )

    def build(self) -> list[dict[str, bool]]:
        """Generate all combinations of data source configurations."""
        grid = self.get_grid_values()
        grid_sources = [src for src in self._sources if len(grid[src]) > 1]
        grid_values = [grid[src] for src in grid_sources]

        if not grid_values:
            return [self._get_single_config()]

        configs = []
        for combo in product(*grid_values):
            config = self._get_single_config()
            for src, val in zip(grid_sources, combo, strict=True):
                config[src] = val
            configs.append(config)
        return configs

    def _get_single_config(self) -> dict[str, bool]:
        grid = self.get_grid_values()
        return {src: grid[src][0] for src in self._sources}

    def get_config_names(self) -> list[str]:
        """Get names for each config for display/logging."""
        configs = self.build()
        names = []
        for config in configs:
            parts = [f"{k}_{v}" for k, v in config.items() if k != "application"]
            if parts:
                names.append("_".join(parts))
            else:
                names.append("app_only")
        return names
