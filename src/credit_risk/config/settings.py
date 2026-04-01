"""Configuration settings using Pydantic.

Two-layer config:
  - AppSettings: paths + runtime config from env vars / .env (12-factor)
  - Settings: ML hyperparams from YAML (versionable, experiment tracking)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache
def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml.

    Walks up from this file until it finds a directory containing
    ``pyproject.toml``.  Falls back to 4-levels-up for legacy structure.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: repo root is 4 levels up from this file
    return current.parent.parent.parent.parent


PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# App-level config: paths, env vars (12-factor, never committed)
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Application-level configuration.

    Loaded from environment variables (``CREDIT_RISK_*``) or ``.env`` file.
    Override any field at runtime: ``CREDIT_RISK_DATA_PATH=/data uv run ...``
    """

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_RISK_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    data_path: Path = Field(default=PROJECT_ROOT / "data")
    models_path: Path = Field(default=PROJECT_ROOT / "models")
    logs_path: Path = Field(default=PROJECT_ROOT / "logs")
    output_path: Path = Field(default=PROJECT_ROOT / "output")


@lru_cache
def get_settings() -> AppSettings:
    """Get application settings (lazy singleton)."""
    return AppSettings()


# ---------------------------------------------------------------------------
# ML-specific config: hyperparams, feature lists (YAML-backed, versionable)
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    target_column: str = "TARGET"
    id_column: str = "SK_ID_CURR"
    test_id_column: str = "SK_ID_CURR"
    categorical_threshold: int = 10
    train_size: float = 0.8


class FeatureConfig(BaseModel):
    bureau_agg_features: list[str] = Field(
        default=[
            "SK_ID_BUREAU",
            "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE",
            "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_SUM",
            "AMT_ANNUITY",
        ]
    )
    previous_app_agg_features: list[str] = Field(
        default=[
            "SK_ID_PREV",
            "AMT_ANNUITY",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "AMT_DOWN_PAYMENT",
            "DAYS_DECISION",
            "CNT_PAYMENT",
        ]
    )
    installments_agg_features: list[str] = Field(
        default=["NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION", "AMT_INSTALMENT", "AMT_PAYMENT"]
    )
    pos_cash_agg_features: list[str] = Field(
        default=[
            "MONTHS_BALANCE",
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "SK_DPD",
            "SK_DPD_DEF",
        ]
    )
    credit_card_agg_features: list[str] = Field(
        default=[
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_OTHER_CURRENT",
            "AMT_DRAWINGS_POS_CURRENT",
            "AMT_INST_MIN_REGULARITY",
            "AMT_PAYMENT_CURRENT",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "AMT_RECEIVABLE_PRINCIPAL",
            "AMT_RECIVABLE",
            "AMT_TOTAL_RECEIVABLE",
            "CNT_DRAWINGS_ATM_CURRENT",
            "CNT_DRAWINGS_CURRENT",
            "CNT_DRAWINGS_OTHER_CURRENT",
            "CNT_DRAWINGS_POS_CURRENT",
            "CNT_INSTALMENT_MATURE_CUM",
            "SK_DPD",
            "SK_DPD_DEF",
        ]
    )
    bureau_balance_agg_features: list[str] = Field(default=["MONTHS_BALANCE", "STATUS"])


class ModelConfig(BaseModel):
    model_type: Literal["lightgbm", "sklearn"] = "lightgbm"
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 100


class InterpretConfig(BaseModel):
    shap_background_samples: int = 1000
    shap_n_samples: int = 100


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings(BaseModel):
    """ML pipeline configuration.

    Loaded from YAML or defaults.  Committed to git for experiment tracking.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    interpret: InterpretConfig = Field(default_factory=InterpretConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load ML settings from YAML or return defaults."""
    if config_path is None:
        return Settings()
    return Settings.from_yaml(Path(config_path))
