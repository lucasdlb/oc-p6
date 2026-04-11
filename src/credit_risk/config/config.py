"""Configuration loader and global config instance."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from credit_risk.config.models import Config


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"


@lru_cache
def load_config() -> Config:
    """Load configuration from TOML files.

    Combines mode-specific config (debug/dev/prod.toml) with static data.toml.

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If mode config not found
    """
    mode = os.getenv("RUN_MODE", "prod")

    mode_path = CONFIG_DIR / f"{mode}.toml"
    if not mode_path.exists():
        raise FileNotFoundError(
            f"No config found for mode '{mode}' at {mode_path}. Available modes: debug, dev, prod"
        )

    with open(mode_path, "rb") as f:
        mode_config = tomllib.load(f)

    data_path = CONFIG_DIR / "data.toml"
    with open(data_path, "rb") as f:
        data_config = tomllib.load(f)

    full_config: dict[str, Any] = {**mode_config, **data_config}

    return Config(**full_config)


cfg = load_config()


def reload_config() -> Config:
    """Reload configuration (clears cache).

    Useful during testing or when config changes.
    """
    load_config.cache_clear()
    return load_config()
