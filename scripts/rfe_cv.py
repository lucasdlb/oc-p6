#!/usr/bin/env python
"""Unified RFE-CV feature selection for all tables.

Proper ML pipeline:
1. Split data into train/test (test held out until final evaluation)
2. Use RFE-CV on training set only to select best features
3. Evaluate final model on held-out test set

Usage:
    RUN_MODE=debug uv run python scripts/rfe_cv.py --table application
    RUN_MODE=dev   uv run python scripts/rfe_cv.py --table application
    uv run python scripts/rfe_cv.py --table application  # defaults to prod
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import mlflow
import polars as pl

from credit_risk.config import cfg
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoding import CategoricalEncoder
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.features.aggregator import DataAggregator
from credit_risk.features.store import FeatureStore
from credit_risk.features.transformer import DataTransformer
from credit_risk.models.cross_validator import LGBMFactory
from credit_risk.models.feature_selector import BackwardFeatureSelector
from credit_risk.models.final_model import FinalModelTrainer
from credit_risk.models.importance import get_importance_class
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.threshold_selector import CVThresholdSelector

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TABLE_CONFIGS = {
    "application": {"experiment": "application_rfe_cv", "has_encoding": True},
    "bureau": {"experiment": "bureau_rfe_cv"},
    "bureau_balance": {"experiment": "bureau_balance_rfe_cv"},
    "previous_application": {"experiment": "previous_application_rfe_cv"},
    "pos_cash_balance": {"experiment": "pos_cash_balance_rfe_cv"},
    "credit_card_balance": {"experiment": "credit_card_balance_rfe_cv"},
    "installments_payments": {"experiment": "installments_payments_rfe_cv"},
}

parser = argparse.ArgumentParser(description="RFE-CV feature selection")
parser.add_argument("--table", type=str, choices=list(TABLE_CONFIGS.keys()), default="application")
args = parser.parse_args()

table_config = TABLE_CONFIGS[args.table]
run_mode = cfg.run.mode
logger.info(f"Running in mode: {run_mode}, table: {args.table}")

sample_frac = cfg.run.sample_fraction
if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")

splitter = TrainTestCVSplitter(
    test_size=cfg.splitter.test_size,
    n_splits=cfg.splitter.n_splits,
    random_state=cfg.splitter.random_state,
    stratify=True,
)
ranking_metrics = ClassificationRankingMetrics(roc_auc=True)
model_factory = LGBMFactory()
threshold_selector = CVThresholdSelector(
    splitter=splitter, model_factory=model_factory, metric="f1"
)

ImportanceClass = get_importance_class(cfg.importance.method)
importance_strategy = ImportanceClass()

if cfg.mlflow.enabled:
    mlflow.set_tracking_uri(f"sqlite:///{cfg.output.mlflow_db_path}")
    mlflow.set_experiment(table_config["experiment"])

    if mlflow.active_run():
        mlflow.end_run()

    run_name = f"{run_mode}_{args.table}"
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({"run_mode": run_mode, "table": args.table})
    mlflow.log_params(cfg.model.model_dump())

    mlflow.log_dict(cfg.model_dump(), "config.json")

loader = PLLazyDataLoader()
labels = loader.load_labels().collect()

df = loader.load(args.table)
df = df.collect() if hasattr(df, "collect") else df

if sample_frac < 1.0:
    df = df.sample(fraction=sample_frac, seed=cfg.run.random_state)

logger.info(f"{args.table}: {df.height} rows, {df.width} cols")

cleaner = DataCleaner()
imputer = DataImputer()
aggregator = DataAggregator()
transformer = DataTransformer()

df = cleaner.clean(df, args.table)
df = imputer.impute(df, args.table)
df = aggregator.aggregate(df.lazy(), args.table, method="detailed").collect()
df = transformer.transform(df, table=args.table)

if table_config.get("has_encoding"):
    encoder = CategoricalEncoder()
    df = encoder.fit_transform(df)
    logger.info(f"Encoded: {df.height} rows, {df.width} cols")

main_df = labels.join(df, on="SK_ID_CURR", how="inner")
logger.info(f"Dataset: {main_df.height} rows")

non_numeric = [c for c in main_df.columns if main_df.schema[c] == pl.String]
if non_numeric:
    main_df = main_df.drop(non_numeric)

target_col = cfg.data.target.column
id_col = cfg.data.target.id_column

y_full = main_df.select(target_col).to_numpy().ravel()

suffix_cols = [c for c in main_df.columns if c.endswith("_right")]
if suffix_cols:
    main_df = main_df.drop(suffix_cols)

feature_cols = [c for c in main_df.columns if c not in [target_col, id_col]]

X_full = (
    main_df.select(feature_cols)
    .to_pandas()
    # .fillna(0.0)
    # .replace([float("inf"), float("-inf")], 0.0)
    .values
)
y_full = main_df.select(target_col).to_numpy().ravel()

logger.info(f"Features: {len(feature_cols)}")
logger.info(f"Total samples: {X_full.shape[0]}")

X_train, X_test, y_train, y_test = splitter.split_train_test(X_full, y_full)

logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

model_params = cfg.model.model_dump()

logger.info(
    f"RFE-CV backward selection (min {cfg.selection.min_features} features, "
    f"remove {cfg.selection.nb_remove_features} per step, importance: {cfg.importance.method})..."
)
selector = BackwardFeatureSelector(
    splitter=splitter,
    metrics=ranking_metrics,
    model_factory=model_factory,
    importance_strategy=importance_strategy,
    selection_metric_name="roc_auc",
    min_features=cfg.selection.min_features,
    tolerance=cfg.selection.tolerance,
    nb_remove_features=cfg.selection.nb_remove_features,
    verbose=True,
)
best_features, best_cv_result = selector.eliminate(
    X_train, y_train, model_params, feature_names=feature_cols
)

if cfg.mlflow.enabled:
    mlflow.log_metrics(
        {
            "cv_best_roc_auc": best_cv_result.mean_scores.get("roc_auc", 0),
            "cv_best_roc_auc_std": best_cv_result.std_scores.get("roc_auc", 0),
            "best_n_features": len(best_features),
        }
    )

logger.info(f"Best feature set: {len(best_features)} features")
logger.info(f"Best features: {best_features}")

best_indices = [feature_cols.index(f) for f in best_features]
X_train_best = X_train[:, best_indices]
X_test_best = X_test[:, best_indices]

trainer = FinalModelTrainer(
    model_factory, threshold_selector, importance_strategy=importance_strategy
)
result = trainer.train_and_evaluate(
    X_train_best, y_train, X_test_best, y_test, best_features, model_params
)

if cfg.mlflow.enabled:
    mlflow.log_metrics(
        {
            "test_roc_auc": result.test_roc_auc,
            "test_f1": result.test_f1,
            "test_recall": result.test_recall,
            "test_precision": result.test_precision,
            "test_accuracy": result.test_accuracy,
            "optimal_threshold": result.optimal_threshold,
        }
    )

feature_store = FeatureStore(root=cfg.output.features_path)
feature_store.save(
    name=f"{args.table}_{run_mode}",
    features=best_features,
    meta={
        "table": args.table,
        "run_mode": run_mode,
        "cv_roc_auc": best_cv_result.mean_scores.get("roc_auc", 0),
        "cv_roc_auc_std": best_cv_result.std_scores.get("roc_auc", 0),
        "test_roc_auc": result.test_roc_auc,
        "test_f1": result.test_f1,
        "model_params": model_params,
    },
)
logger.info(
    f"Saved features to FeatureStore: {args.table}_{run_mode} ({len(best_features)} features)"
)

if cfg.mlflow.enabled:
    import json
    import tempfile

    features_data = {
        "table": args.table,
        "features": best_features,
        "n_features": len(best_features),
        "cv_roc_auc": best_cv_result.mean_scores.get("roc_auc", 0),
        "test_roc_auc": result.test_roc_auc,
    }
    features_path = os.path.join(tempfile.gettempdir(), f"{args.table}_features.json")
    with open(features_path, "w") as f:
        json.dump(features_data, f, indent=2)
    mlflow.log_artifact(features_path)

final_model = model_factory.create(**model_params)
final_model.fit(X_train_best, y_train)
if cfg.mlflow.enabled:
    mlflow.sklearn.log_model(final_model, "model")
    mlflow.end_run()

logger.info("Pipeline complete")
