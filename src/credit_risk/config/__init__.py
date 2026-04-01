"""Configuration package.

Exports:
  - PROJECT_ROOT: Dynamically resolved project root directory
  - AppSettings: Paths + env-var config (12-factor)
  - Settings: ML hyperparams config (YAML-backed)
  - get_settings(): Cached AppSettings singleton
  - load_settings(): Load Settings from YAML
"""

from __future__ import annotations

from credit_risk.config.settings import (
    PROJECT_ROOT,
    AppSettings,
    Settings,
    get_settings,
    load_settings,
)

__all__ = [
    "PROJECT_ROOT",
    "AppSettings",
    "Settings",
    "get_settings",
    "load_settings",
]
