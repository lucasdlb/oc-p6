"""Model registry for grid search configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

REGISTRY: dict[str, "ModelGridConfig"] = {}

PRESETS: dict[str, list[str]] = {
    "debug": ["LogisticRegression"],
    "fast": ["LogisticRegression", "RandomForest", "LightGBM"],
    "full": ["LogisticRegression", "RandomForest", "LightGBM"],
}


@dataclass
class ModelGridConfig:
    """Configuration for a model grid search entry."""

    estimator_name: str
    estimator_factory: Callable[..., Any]
    param_grid: dict[str, list[Any]]


def _create_logistic_regression() -> Any:
    """Create LogisticRegression with pipeline (scaling)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])


def _create_random_forest() -> Any:
    """Create RandomForest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier()


def _create_lightgbm() -> Any:
    """Create LightGBM classifier."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier()


def _init_registry() -> None:
    """Initialize the model registry with available estimators."""
    global REGISTRY

    REGISTRY = {
        "LogisticRegression": ModelGridConfig(
            estimator_name="LogisticRegression",
            estimator_factory=_create_logistic_regression,
            param_grid={
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs"],
                "clf__class_weight": ["balanced"],
                "clf__max_iter": [1000],
            },
        ),
        "RandomForest": ModelGridConfig(
            estimator_name="RandomForest",
            estimator_factory=_create_random_forest,
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "class_weight": ["balanced"],
            },
        ),
        "LightGBM": ModelGridConfig(
            estimator_name="LightGBM",
            estimator_factory=_create_lightgbm,
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31],
                "class_weight": ["balanced"],
            },
        ),
    }


def get_configs(preset: str) -> list[ModelGridConfig]:
    """Get list of ModelGridConfig for a preset.

    Args:
        preset: Preset name (debug, fast, full)

    Returns:
        List of ModelGridConfig instances

    Raises:
        KeyError: If preset not found
    """
    if not REGISTRY:
        _init_registry()

    if preset not in PRESETS:
        raise KeyError(f"Preset '{preset}' not found. Available: {list(PRESETS.keys())}")

    keys = PRESETS[preset]
    return [REGISTRY[k] for k in keys]


def get_all_configs() -> list[ModelGridConfig]:
    """Get all available ModelGridConfig."""
    if not REGISTRY:
        _init_registry()
    return list(REGISTRY.values())


def trim_grid(
    configs: list[ModelGridConfig],
    max_values: int = 1,
) -> list[ModelGridConfig]:
    """Trim grid search config to keep only first N values per param.

    Useful for debug runs where full grid would take too long.

    Args:
        configs: List of ModelGridConfig
        max_values: Maximum values to keep per parameter

    Returns:
        Trimmed list of ModelGridConfig
    """
    trimmed = []
    for cfg in configs:
        trimmed.append(
            ModelGridConfig(
                estimator_name=cfg.estimator_name,
                estimator_factory=cfg.estimator_factory,
                param_grid={k: v[:max_values] for k, v in cfg.param_grid.items()},
            )
        )
    return trimmed


# Initialize registry on module load
_init_registry()
