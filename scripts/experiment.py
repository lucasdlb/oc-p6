#!/usr/bin/env python
"""Cross-validation experiment on a single table without RFE or final model training.

Loads one table, processes it, and runs CV to get reliable metrics.

Usage:
    RUN_MODE=debug uv run python scripts/experiment.py --table application
    RUN_MODE=dev   uv run python scripts/experiment.py --table bureau
    uv run python scripts/experiment.py  # defaults to application, prod
"""

from __future__ import annotations

import argparse
import logging
import warnings

import mlflow
import numpy as np

from credit_risk.config import cfg
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoding import PolarsOneHotEncoder
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.features.aggregator import DataAggregator
from credit_risk.features.transformer import DataTransformer
from credit_risk.models.cross_validator import CrossValidator, LGBMFactory
from credit_risk.models.metrics import ClassificationRankingMetrics

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TABLE_CONFIGS = {
    "application": {"experiment": "application_cv", "has_encoding": True},
    "bureau": {"experiment": "bureau_cv"},
    "bureau_balance": {"experiment": "bureau_balance_cv"},
    "previous_application": {"experiment": "previous_application_cv"},
    "pos_cash_balance": {"experiment": "pos_cash_balance_cv"},
    "credit_card_balance": {"experiment": "credit_card_balance_cv"},
    "installments_payments": {"experiment": "installments_payments_cv"},
}

parser = argparse.ArgumentParser(description="Run CV experiment on a single table")
parser.add_argument("--table", type=str, default="application", choices=list(TABLE_CONFIGS.keys()))
args = parser.parse_args()

table = args.table
table_config = TABLE_CONFIGS[table]
run_mode = cfg.run.mode
logger.info(f"Running experiment on table: {table}, mode: {run_mode}")

if cfg.mlflow.enabled:
    mlflow.set_tracking_uri(f"sqlite:///{cfg.output.mlflow_db_path}")
    mlflow.set_experiment(table_config["experiment"])
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.start_run(run_name=f"{run_mode}_{table}_cv")
    mlflow.log_params({"run_mode": run_mode, "table": table})
    mlflow.log_dict(cfg.model_dump(), "config.json")

sample_frac = cfg.run.sample_fraction
if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")

logger.info("Loading data...")
loader = PLLazyDataLoader()
labels = loader.load_labels().collect()

df = loader.load(table)
df = df.collect() if hasattr(df, "collect") else df

if sample_frac < 1.0:
    df = df.sample(fraction=sample_frac, seed=cfg.run.random_state)

logger.info(f"{table}: {df.height} rows, {df.width} cols")

cleaner = DataCleaner()
imputer = DataImputer()
aggregator = DataAggregator()
transformer = DataTransformer()

df = cleaner.clean(df, table)
df = imputer.impute(df, table)
df = aggregator.aggregate(df.lazy(), table, method="detailed").collect()
df = transformer.transform(df, table=table)

if table_config.get("has_encoding"):
    # encoder = CategoricalEncoder()
    encoder = PolarsOneHotEncoder(max_categories=20)
    df = encoder.fit_transform(df)
    logger.info(f"Encoded: {df.height} rows, {df.width} cols")

target_col = cfg.data.target.column
id_col = cfg.data.target.id_column

feature_cols = [
    c for c in df.columns if c not in [id_col, target_col, "SK_ID_BUREAU", "SK_ID_PREV"]
]
logger.info(f"Using {len(feature_cols)} features")

X = df.select(feature_cols).to_pandas()
y = df.select(target_col).to_numpy().ravel()

# X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
X = X.to_numpy(dtype=np.float64)

logger.info(f"Data shape: {X.shape}, Features: {len(feature_cols)}")

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=cfg.splitter.n_splits, shuffle=True, random_state=cfg.splitter.random_state
)
metrics = ClassificationRankingMetrics(roc_auc=True)
factory = LGBMFactory()

logger.info(f"Running {cfg.splitter.n_splits}-fold cross-validation...")

validator = CrossValidator(
    splitter=skf,
    metrics=metrics,
    model_factory=factory,
)

model_params = cfg.model.model_dump()
model_params["verbose"] = -1

result = validator.validate(X, y, model_params=model_params)

logger.info("=" * 60)
logger.info(f"CV Results ({cfg.splitter.n_splits} folds, {len(feature_cols)} features):")
logger.info(f"  ROC AUC: {result.mean_scores['roc_auc']:.4f} ± {result.std_scores['roc_auc']:.4f}")
logger.info("=" * 60)

for fold_idx, fold_score in enumerate(result.fold_scores):
    logger.info(f"Fold {fold_idx + 1}: ROC AUC = {fold_score['roc_auc']:.4f}")

if cfg.mlflow.enabled:
    mlflow.log_metrics(
        {
            "cv_roc_auc": result.mean_scores["roc_auc"],
            "cv_roc_auc_std": result.std_scores["roc_auc"],
            "n_folds": cfg.splitter.n_splits,
            "n_features": len(feature_cols),
        }
    )
    mlflow.end_run()

logger.info("Experiment complete")
