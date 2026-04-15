"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

    max_depth: int = Field(gt=0, default=5)
    n_estimators: int = Field(gt=0, default=500)
    learning_rate: float = Field(gt=0, default=0.02)
    num_leaves: int = Field(gt=0, default=31)
    min_child_samples: int = Field(gt=0, default=20)
    subsample: float = Field(ge=0.0, le=1.0, default=0.8)
    colsample_bytree: float = Field(ge=0.0, le=1.0, default=0.8)
    reg_alpha: float = Field(default=0.1)
    reg_lambda: float = Field(default=0.1)
    n_jobs: int = Field(default=-1)
    class_weight: str = Field(default="balanced")
    verbose: int = Field(default=-1)


class SelectionConfig(BaseModel):
    """Feature selection configuration."""

    min_features: int = Field(gt=0, default=20)
    tolerance: float = Field(gt=0, default=0.005)
    nb_remove_features: float = Field(default=0.3)


class ImportanceConfig(BaseModel):
    """Feature importance strategy configuration."""

    method: Literal["inner", "forest", "statistical", "permutation", "shap"] = "inner"


class MLFlowConfig(BaseModel):
    """MLflow tracking configuration."""

    enabled: bool = True
    experiment_name: str = "experiment"


class InterpretConfig(BaseModel):
    """SHAP explainer configuration."""

    shap_background_samples: int = 1000
    shap_n_samples: int = 100


class SearchConfig(BaseModel):
    """Grid search configuration."""

    preset: str = "fast"
    max_grid_values: int = Field(gt=0, default=3)


class TuningConfig(BaseModel):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int = Field(default=50, ge=1)
    timeout: int | None = None
    study_name: str = "credit_risk_tuning"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_jobs: int = Field(default=1, ge=1)
    pruner: Literal["median", "hyperband", "none"] = "none"
    models: list[str] = Field(default_factory=lambda: ["lgbm"])


class ResamplingConfig(BaseModel):
    """Resampling configuration for handling imbalanced data."""

    enabled: bool = False
    method: Literal["smote", "over", "under", "none"] | float = "smote"
    sampling_strategy: str | float = "minority"
    k_neighbors: int = 5
    random_state: int = 42


class OutputConfig(BaseModel):
    """Output paths for artifacts, models, mlflow."""

    models_dir: str = "models"
    features_dir: str = "artifacts/features"
    mlflow_db: str = "mlflow.db"

    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir

    @property
    def features_path(self) -> Path:
        return PROJECT_ROOT / self.features_dir

    @property
    def mlflow_db_path(self) -> Path:
        return PROJECT_ROOT / self.mlflow_db


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


class DataSourcesConfig(BaseModel):
    """Configuration for which data sources to use in experiments."""

    application: bool = True
    bureau: bool = True
    bureau_balance: bool = True
    previous_application: bool = True
    pos_cash_balance: bool = True
    installments_payments: bool = True
    credit_card_balance: bool = True

    def is_enabled(self, source: str) -> bool:
        """Check if a data source is enabled."""
        return getattr(self, source, False)

    def get_enabled_sources(self) -> list[str]:
        """Get list of enabled source names."""
        return [k for k, v in self.model_dump().items() if v is True]


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

    run: RunConfig = Field(default_factory=RunConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    importance: ImportanceConfig = Field(default_factory=ImportanceConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    interpret: InterpretConfig = Field(default_factory=InterpretConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    enabled_tables: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
