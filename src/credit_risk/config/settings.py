from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache
def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = _find_project_root()


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Environment variables override defaults:
        CREDIT_RISK_DATA_PATH=/custom/path
        CREDIT_RISK_MODELS_PATH=/custom/models
        etc.

    Also supports .env file at project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_RISK_",
        env_file=PROJECT_ROOT / ".env",
    )

    data_path: Path = Field(default=PROJECT_ROOT / "data")
    models_path: Path = Field(default=PROJECT_ROOT / "models")
    logs_path: Path = Field(default=PROJECT_ROOT / "logs")
    output_path: Path = Field(default=PROJECT_ROOT / "output")
    notebooks_path: Path = Field(default=PROJECT_ROOT / "notebooks")


@lru_cache
def get_settings() -> Settings:
    """Get application settings (lazy singleton)."""
    return Settings()
