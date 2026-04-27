"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent


PROJECT_ROOT = _find_project_root()


class RunConfig(BaseModel):
    """Runtime execution mode configuration."""

    mode: Literal["debug", "dev", "prod"] = "prod"
    sample_fraction: float = Field(ge=0.0, le=1.0, default=1.0)
    random_state: int = Field(default=42)


class SplitterConfig(BaseModel):
    """Train/test split configuration."""

    test_size: float = Field(ge=0.0, le=1.0, default=0.2)
    n_splits: int = Field(gt=0, default=5)
    cv_random_state: int = Field(default=42)
    test_random_state: int = Field(default=42)
    stratify: bool = Field(default=True)
    shuffle: bool = Field(default=True)


class ModelConfig(BaseModel):
    """Model hyperparameters configuration."""

    model_type: str
    x_transform: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)


class SelectionConfig(BaseModel):
    """Feature selection configuration."""

    min_features: int = Field(gt=0)
    tolerance: float = Field(gt=0)
    nb_remove_features: float

    @classmethod
    def from_toml(cls) -> "SelectionConfig":
        """Load SelectionConfig from mode-specific TOML."""
        import os
        import tomllib

        mode = os.getenv("RUN_MODE", "prod")
        from credit_risk.config.config import CONFIG_DIR

        with open(CONFIG_DIR / f"{mode}.toml", "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("selection", {}))


class ImportanceConfig(BaseModel):
    """Feature importance strategy configuration."""

    method: Literal["inner", "forest", "statistical", "permutation", "shap"] = "inner"

    @classmethod
    def from_toml(cls) -> "ImportanceConfig":
        """Load ImportanceConfig from mode-specific TOML."""
        import os
        import tomllib

        mode = os.getenv("RUN_MODE", "prod")
        from credit_risk.config.config import CONFIG_DIR

        with open(CONFIG_DIR / f"{mode}.toml", "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("importance", {}))


class InterpretConfig(BaseModel):
    """SHAP explainer configuration."""

    shap_background_samples: int = 1000
    shap_n_samples: int = 100

    @classmethod
    def from_toml(cls) -> "InterpretConfig":
        """Load InterpretConfig from mode-specific TOML."""
        import os
        import tomllib

        mode = os.getenv("RUN_MODE", "prod")
        from credit_risk.config.config import CONFIG_DIR

        with open(CONFIG_DIR / f"{mode}.toml", "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("interpret", {}))


class TuningConfig(BaseModel):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int = Field(default=50, ge=1)
    timeout: int | None = None
    study_name: str = "credit_risk_tuning"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_jobs: int = Field(default=1, ge=1)
    pruner: Literal["median", "hyperband", "none"] = "none"
    models: list[str] = Field(default_factory=lambda: ["lgbm"])

    @classmethod
    def from_toml(cls) -> "TuningConfig":
        """Load TuningConfig from mode-specific TOML."""
        import os
        import tomllib

        mode = os.getenv("RUN_MODE", "prod")
        from credit_risk.config.config import CONFIG_DIR

        with open(CONFIG_DIR / f"{mode}.toml", "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("tuning", {}))


class ResamplingConfig(BaseModel):
    """Resampling configuration for handling imbalanced data."""

    enabled: bool = False
    method: Literal["smote", "over", "under", "none"] | float = "smote"
    sampling_strategy: str | float = "minority"
    k_neighbors: int = 5
    random_state: int = 42

    @classmethod
    def from_toml(cls) -> "ResamplingConfig":
        """Load ResamplingConfig from mode-specific TOML."""
        import os
        import tomllib

        mode = os.getenv("RUN_MODE", "prod")
        from credit_risk.config.config import CONFIG_DIR

        with open(CONFIG_DIR / f"{mode}.toml", "rb") as f:
            data = tomllib.load(f)
        return cls(**data.get("resampling", {}))


class OutputConfig(BaseModel):
    """Output paths for artifacts, models, mlflow."""

    models_dir: str = "models"
    features_dir: str = "artifacts/features"
    mlflow_db: str = "mlflow.db"
    mlflow_db_type: Literal["sqlite", "postgresql", "mysql"] = "sqlite"

    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir

    @property
    def features_path(self) -> Path:
        return PROJECT_ROOT / self.features_dir

    @property
    def mlflow_db_path(self) -> Path:
        return PROJECT_ROOT / self.mlflow_db

    def mlflow_tracking_uri(self) -> str:
        """Build MLflow tracking URI based on db type."""
        if self.mlflow_db_type == "sqlite":
            return f"sqlite:///{self.mlflow_db_path}"
        elif self.mlflow_db_type == "postgresql":
            return f"postgresql:///{self.mlflow_db}"
        elif self.mlflow_db_type == "mysql":
            return f"mysql:///{self.mlflow_db}"
        return f"sqlite:///{self.mlflow_db_path}"


# -----------------------------------------------------------------------------
# Data config (static, loaded from data.toml)
# -----------------------------------------------------------------------------


class DataSourceFiles(BaseModel):
    """Data source file paths."""

    application: str = "application_train.csv"
    bureau: str = "bureau.csv"
    bureau_balance: str = "bureau_balance.csv"
    previous_application: str = "previous_application.csv"
    pos_cash: str = "POS_CASH_balance.csv"
    installments: str = "installments_payments.csv"
    credit_card: str = "credit_card_balance.csv"


class TargetConfig(BaseModel):
    """Target variable configuration."""

    column: str = "TARGET"
    id_column: str = "SK_ID_CURR"


class FeaturesConfig(BaseModel):
    """Feature configuration."""

    drop_always: list[str] = Field(
        default_factory=lambda: ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
    )
    categorical: list[str] = Field(default_factory=list)
    ext_source: list[str] = Field(
        default_factory=lambda: ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    )

    bureau_agg_features: list[str] = Field(default_factory=list)
    previous_app_agg_features: list[str] = Field(default_factory=list)
    installments_agg_features: list[str] = Field(default_factory=list)
    pos_cash_agg_features: list[str] = Field(default_factory=list)
    credit_card_agg_features: list[str] = Field(default_factory=list)
    bureau_balance_agg_features: list[str] = Field(default_factory=list)


class TableConfig(BaseModel):
    """Configuration for a single table's processing steps.

    Each table specifies which class implements each step.
    The composite resolves the class name from the appropriate registry.
    """

    include: bool = True
    cleaner: str = "RawCleaner"
    imputer: str = "RawImputer"
    aggregator: str = "NoOpAggregator"
    transformer: str = "ApplicationTransformer"
    encoder: str = "PolarsOneHotEncoder"


class DataConfig(BaseModel):
    """Data configuration (static, mode-independent)."""

    data_dir: str = "data"
    output_dir: str = "output"
    target: TargetConfig = Field(default_factory=TargetConfig)
    sources: DataSourceFiles = Field(default_factory=DataSourceFiles)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    application: TableConfig = Field(default_factory=TableConfig)
    bureau: TableConfig = Field(default_factory=TableConfig)
    bureau_balance: TableConfig = Field(default_factory=TableConfig)
    previous_application: TableConfig = Field(default_factory=TableConfig)
    pos_cash: TableConfig = Field(default_factory=TableConfig)
    installments: TableConfig = Field(default_factory=TableConfig)
    credit_card: TableConfig = Field(default_factory=TableConfig)


# -----------------------------------------------------------------------------
# Full config (mode + data)
# -----------------------------------------------------------------------------


class Config(BaseModel):
    """Full configuration combining runtime mode and static data config."""

    run: RunConfig = Field(default_factory=RunConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model: Optional[ModelConfig] = None
    selection: Optional[SelectionConfig] = None
    tuning: Optional[TuningConfig] = None
    resampling: Optional[ResamplingConfig] = None
    interpret: Optional[InterpretConfig] = None
    importance: Optional[ImportanceConfig] = None
