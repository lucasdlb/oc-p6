#!/usr/bin/env python
"""Optuna hyperparameter tuning across ALL tables.

Loads, cleans, imputes, aggregates ALL tables, joins on SK_ID_CURR,
then tunes multiple models using Optuna sequential optimization.

Usage:
    RUN_MODE=debug uv run python scripts/tune.py
    RUN_MODE=dev   uv run python scripts/tune.py
    uv run python scripts/tune.py  # defaults to prod
"""

from __future__ import annotations

import logging
import warnings

import mlflow
import numpy as np
import polars as pl

from credit_risk.config import cfg
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoding import CategoricalEncoder
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.features.aggregator import DataAggregator
from credit_risk.features.transformer import DataTransformer
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.tuner import ManyModelOptunaTuner

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TABLES = [
    "application",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "credit_card_balance",
    "installments_payments",
]

USE_SELECTED_FEATURES = True  # Set to False to use all features
run_mode = cfg.run.mode
logger.info(f"Running tuning in mode: {run_mode}")

if cfg.mlflow.enabled:
    mlflow.set_tracking_uri(f"sqlite:///{cfg.output.mlflow_db_path}")
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.start_run(run_name=f"{run_mode}_tuning")
    mlflow.log_params({"run_mode": run_mode, "n_trials": cfg.tuning.n_trials})
    mlflow.log_param("models", ",".join(cfg.tuning.models))
    mlflow.log_dict(cfg.model_dump(), "config.json")

models_to_tune = cfg.tuning.models
logger.info(f"Models to tune: {models_to_tune}, Trials: {cfg.tuning.n_trials}")

splitter = TrainTestCVSplitter(
    test_size=cfg.splitter.test_size,
    n_splits=cfg.splitter.n_splits,
    random_state=cfg.splitter.random_state,
    stratify=True,
)
metrics = ClassificationRankingMetrics(roc_auc=True)

logger.info(f"CV: {cfg.splitter.n_splits} splits, Trials: {cfg.tuning.n_trials}")

sample_frac = cfg.run.sample_fraction
if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")

logger.info("Loading data...")
loader = PLLazyDataLoader()
labels = loader.load_labels()

cleaner = DataCleaner()
imputer = DataImputer()
aggregator = DataAggregator()
transformer = DataTransformer()

features_list = []
for table in TABLES:
    logger.info(f"Processing table: {table}")

    df = loader.load(table)
    df = df.collect() if hasattr(df, "collect") else df

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac)

    logger.info(f"  Loaded: {df.height} rows, {df.width} cols")

    df = cleaner.clean(df, table, method=cfg.processing.cleaning)
    df = imputer.impute(df, table, method=cfg.processing.imputation)

    df = aggregator.aggregate(df.lazy(), table, method=cfg.processing.aggregation).collect()

    df = transformer.transform(df, table=table)

    id_cols = [c for c in df.columns if c.startswith("SK_ID")]
    feature_cols = [c for c in df.columns if c not in id_cols + ["TARGET"]]
    features_list.append(df.select(["SK_ID_CURR"] + feature_cols))

logger.info(f"Joining {len(features_list)} tables with labels...")

combined = labels.lazy()
for i, df in enumerate(features_list):
    combined = combined.join(df.lazy(), on="SK_ID_CURR", how="left", suffix=f"_{i}")

combined = combined.collect()

logger.info(f"Combined: {combined.height} rows, {combined.width} cols")

if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of combined data")
    combined = combined.sample(fraction=sample_frac, seed=cfg.run.random_state)
    logger.info(f"Sampled: {combined.height} rows")

target_col = cfg.data.target.column
id_col = cfg.data.target.id_column

non_numeric = [c for c in combined.columns if combined.schema[c] == pl.String]
if non_numeric:
    logger.info(f"Encoding {len(non_numeric)} categorical columns")
    encoder = CategoricalEncoder()
    combined = encoder.fit_transform(combined)
    logger.info(f"Encoded: {combined.height} rows, {combined.width} cols")

if USE_SELECTED_FEATURES:
    from credit_risk.features.store import FeatureStore

    store = FeatureStore(root=cfg.output.features_path)
    all_selected, _ = store.load_tables(TABLES, suffix=f"_{run_mode}")
    all_selected = list(set(all_selected))
    available_cols = set(combined.columns) - {target_col, id_col}
    feature_cols = [c for c in all_selected if c in available_cols]
else:
    feature_cols = [c for c in combined.columns if c not in [target_col, id_col]]

logger.info(f"Data shape: {X.shape}, Features: {len(feature_cols)}")

logger.info("Starting tuning...")
tuner = ManyModelOptunaTuner(
    splitter=splitter,
    metrics=metrics,
    tuning_config=cfg.tuning,
    mlflow_logging=cfg.mlflow.enabled,
)

logger.info(f"Tuning models sequentially: {models_to_tune}")
results = tuner.optimize_sequential(X, y, models_to_tune)

logger.info("Training final model...")
final_model, best_model_name, all_results = tuner.get_best_model(X, y)

if cfg.mlflow.enabled:
    mlflow.log_dict(cfg.model_dump(), "config.json")
    for model_name, result in all_results.items():
        mlflow.log_metric(f"{model_name}_roc_auc", result["best_value"])
    # Log actual tuned params from tuner results
    mlflow.log_params(all_results[best_model_name]["best_params"])
    mlflow.log_metric("best_roc_auc", all_results[best_model_name]["best_value"])
    mlflow.log_param("best_model", best_model_name)
    mlflow.sklearn.log_model(final_model, "best_model")
    mlflow.end_run()

logger.info(f"Best model: {best_model_name}")

logger.info("=" * 60)
logger.info("All models summary:")
for model_name, result in all_results.items():
    logger.info(f"  {model_name}: ROC AUC = {result['best_value']:.4f}")
logger.info("=" * 60)
