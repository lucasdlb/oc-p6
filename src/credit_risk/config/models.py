"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

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
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """Model hyperparameters configuration."""

    max_depth: int = Field(gt=0)
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    num_leaves: int = Field(gt=0)
    min_child_samples: int = Field(gt=0)
    subsample: float = Field(ge=0.0, le=1.0)
    colsample_bytree: float = Field(ge=0.0, le=1.0)
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    class_weight: str
    verbose: int


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


class CleanerConfig(BaseModel):
    """Configuration for data cleaning step."""

    method: str = "default"


class ImputerConfig(BaseModel):
    """Configuration for data imputation step."""

    method: str = "default"


class AggregatorConfig(BaseModel):
    """Configuration for feature aggregation step."""

    method: str = "detailed"


class TransformerConfig(BaseModel):
    """Configuration for feature transformation step."""

    encoding: Literal["onehot", "label", "none"] = "onehot"


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
            # For postgresql, expects db name in mlflow_db field
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


class AggregationConfig(BaseModel):
    """Aggregation configuration for a table."""

    include: bool = True
    features: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    """Data configuration (static, mode-independent)."""

    data_dir: str = "data"
    output_dir: str = "output"
    target: TargetConfig = Field(default_factory=TargetConfig)
    sources: DataSourceFiles = Field(default_factory=DataSourceFiles)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    bureau: AggregationConfig = Field(default_factory=AggregationConfig)
    bureau_balance: AggregationConfig = Field(default_factory=AggregationConfig)
    previous_application: AggregationConfig = Field(default_factory=AggregationConfig)
    pos_cash: AggregationConfig = Field(default_factory=AggregationConfig)
    installments: AggregationConfig = Field(default_factory=AggregationConfig)
    credit_card: AggregationConfig = Field(default_factory=AggregationConfig)


# -----------------------------------------------------------------------------
# Full config (mode + data)
# -----------------------------------------------------------------------------


class Config(BaseModel):
    """Full configuration combining runtime mode and static data config."""

    # Always required
    run: RunConfig = Field(default_factory=RunConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Top-level processing step configs
    cleaner: CleanerConfig = Field(default_factory=CleanerConfig)
    imputer: ImputerConfig = Field(default_factory=ImputerConfig)
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)

    # Step-specific — only needed if that step runs
    model: Optional[ModelConfig] = None
    selection: Optional[SelectionConfig] = None
    tuning: Optional[TuningConfig] = None
    resampling: Optional[ResamplingConfig] = None
    interpret: Optional[InterpretConfig] = None
    importance: Optional[ImportanceConfig] = None
