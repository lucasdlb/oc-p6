#!/usr/bin/env python
"""Unified RFE-CV feature selection combining ALL tables.

Proper ML pipeline:
1. Load, clean, impute, aggregate, transform ALL tables
2. Join all features on SK_ID_CURR
3. Encode categorical columns
4. Split data into train/test (test held out until final evaluation)
5. Use RFE-CV on training set only to select best features
6. Evaluate final model on held-out test set

Usage:
    RUN_MODE=debug uv run python scripts/rfe_cv_all.py
    RUN_MODE=dev   uv run python scripts/rfe_cv_all.py
    uv run python scripts/rfe_cv_all.py  # defaults to prod
"""

from __future__ import annotations

import logging
import os
import warnings

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

TABLES = [
    "application",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "credit_card_balance",
    "installments_payments",
]


def main():
    run_mode = cfg.run.mode
    logger.info(f"Running in mode: {run_mode}")

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

    mlflow_active = None

    if cfg.mlflow.enabled:
        import mlflow

        mlflow_active = mlflow

        mlflow.set_tracking_uri(f"sqlite:///{cfg.output.mlflow_db_path}")
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        if mlflow.active_run():
            mlflow.end_run()

        run_name = f"{run_mode}_n_est_{cfg.model.n_estimators}_depth_{cfg.model.max_depth}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                "run_mode": run_mode,
                "sample_fraction": sample_frac,
                "table_count": len(TABLES),
            }
        )
        mlflow.log_params(cfg.model.model_dump())

        mlflow.log_dict(cfg.model_dump(), "config.json")

loader = PLLazyDataLoader()
labels = loader.load_labels()
if sample_frac < 1.0:
    labels = labels.sample(fraction=sample_frac)

cleaner = DataCleaner()
    imputer = DataImputer()
    aggregator = DataAggregator()
    transformer = DataTransformer()

features_list = []
for table in TABLES:
    logger.info(f"Processing table: {table}")

    df = loader.load(table).lazy().join(labels, on="SK_ID_CURR", how="inner")
    df = df.collect() if hasattr(df, "collect") else df

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

    feature_cols = [c for c in combined.columns if c not in [target_col, id_col]]

    X_full = (
        combined.select(feature_cols)
        .to_pandas()
        # .fillna(0.0)
        # .replace([float("inf"), float("-inf")], 0.0)
        .values
    )
    y_full = combined.select(target_col).to_numpy().ravel()

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Total samples: {X_full.shape[0]}")

    X_train, X_test, y_train, y_test = splitter.split_train_test(X_full, y_full)

    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    model_params = cfg.model.model_dump()

    logger.info(
        f"RFE-CV backward selection (min {cfg.selection.min_features} features, "
        f"remove {cfg.selection.nb_remove_features} per step, "
        f"importance: {cfg.importance.method})..."
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

    if mlflow_active:
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

    trainer = FinalModelTrainer(model_factory, threshold_selector)
    result = trainer.train_and_evaluate(
        X_train_best, y_train, X_test_best, y_test, best_features, model_params
    )

    if mlflow_active:
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
        name=f"all_tables_{run_mode}",
        features=best_features,
        meta={
            "tables": TABLES,
            "run_mode": run_mode,
            "cv_roc_auc": best_cv_result.mean_scores.get("roc_auc", 0),
            "cv_roc_auc_std": best_cv_result.std_scores.get("roc_auc", 0),
            "test_roc_auc": result.test_roc_auc,
            "test_f1": result.test_f1,
            "model_params": model_params,
        },
    )
    logger.info(
        f"Saved features to FeatureStore: all_tables_{run_mode} ({len(best_features)} features)"
    )

    if mlflow_active:
        import json
        import tempfile

        features_data = {
            "tables": TABLES,
            "features": best_features,
            "n_features": len(best_features),
            "cv_roc_auc": best_cv_result.mean_scores.get("roc_auc", 0),
            "test_roc_auc": result.test_roc_auc,
        }
        features_path = os.path.join(tempfile.gettempdir(), "all_tables_features.json")
        with open(features_path, "w") as f:
            json.dump(features_data, f, indent=2)
        mlflow.log_artifact(features_path)

        final_model = model_factory.create(**model_params)
        final_model.fit(X_train_best, y_train)
        mlflow.sklearn.log_model(final_model, "model")

    logger.info("Pipeline complete")

    if mlflow_active:
        mlflow.end_run()


if __name__ == "__main__":
    main()
