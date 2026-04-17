"""Configuration loader - each entrypoint calls load_config() explicitly."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from credit_risk.config.models import Config


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(*enable: str) -> "Config":
    """Load configuration from TOML files.

    Usage:
        # Inference only - no optional steps
        cfg = load_config()

        # Training with specific optional configs
        cfg = load_config("tuning", "selection")

        # Full pipeline with all optional steps
        cfg = load_config("tuning", "selection", "resampling", "interpret", "importance")

    The caller declares which optional configs are needed by name.
    Values always come from the TOML files - the caller owns "shape", TOML owns "values".
    """
    from credit_risk.config.models import Config

    mode = os.getenv("RUN_MODE", "prod")
    raw = {
        **_read_toml(CONFIG_DIR / f"{mode}.toml"),
        **_read_toml(CONFIG_DIR / "data.toml"),
    }

    # Only populate optional configs that are explicitly enabled
    # All others remain as None (not in raw dict)
    optional = {"selection", "tuning", "resampling", "interpret", "importance", "model"}
    enabled = set(enable)

    # For enabled configs, extract their section from raw and keep it
    # For disabled configs, ensure they're not in raw (will use model defaults)
    for field in optional:
        if field not in enabled:
            raw.pop(field, None)  # Remove so Config uses its default (None)

    return Config(**raw)
