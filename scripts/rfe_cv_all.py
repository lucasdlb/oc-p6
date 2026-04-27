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

import json
import logging
import os
import tempfile
import warnings

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.data.store import FeatureStore
from credit_risk.models.feature_selector import BackwardFeatureSelector
from credit_risk.models.final_model import FinalModelTrainer
from credit_risk.models.importance import get_importance_class
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.threshold_selector import CVThresholdSelector
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

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
    "pos_cash",
    "installments",
    "credit_card",
]

LOADER_KEYS = {
    "application": "application",
    "bureau": "bureau",
    "bureau_balance": "bureau_balance",
    "previous_application": "previous_application",
    "pos_cash": "pos_cash_balance",
    "installments": "installments_payments",
    "credit_card": "credit_card_balance",
}


def main():
    cfg = load_config("selection", "importance", "model")
    run_mode = cfg.run.mode
    logger.info(f"Running in mode: {run_mode}")

    sample_frac = cfg.run.sample_fraction
    splitter = TrainTestCVSplitter(
        test_size=cfg.splitter.test_size,
        n_splits=cfg.splitter.n_splits,
        cv_random_state=cfg.splitter.random_state,
        stratify=True,
    )
    ranking_metrics = ClassificationRankingMetrics()
    model_factory = get_factory(cfg.model.model_type, cfg.model.x_transform)
    threshold_selector = CVThresholdSelector(
        splitter=splitter, model_factory=model_factory, metric="f1"
    )

    ImportanceClass = get_importance_class(cfg.importance.method)
    importance_strategy = ImportanceClass()

    mlflow_active = None

    if cfg.output:
        import mlflow

        mlflow_active = mlflow

        mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
        mlflow.set_experiment(f"{run_mode}_rfe_cv_all")

        if mlflow.active_run():
            mlflow.end_run()

        run_name = f"{run_mode}_rfe_cv_all"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {
                "run_mode": run_mode,
                "sample_fraction": sample_frac,
                "table_count": len(TABLES),
            }
        )
        mlflow.log_params({"model_type": cfg.model.model_type})
        mlflow.log_params({"x_transform": cfg.model.x_transform})
        mlflow.log_params(cfg.model.params)

        mlflow.log_dict(cfg.model_dump(), "model_config.json")

    loader = PLLazyDataLoader()
    labels = loader.load_labels()
    if sample_frac < 1.0:
        logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")
        labels = labels.collect().sample(fraction=sample_frac).lazy()

    features_list = []
    for table in TABLES:
        logger.info(f"Processing table: {table}")

        df = loader.load(LOADER_KEYS[table]).join(labels, on="SK_ID_CURR", how="inner")
        df = df.collect()

        logger.info(f"  Loaded: {df.height} rows, {df.width} cols")

        table_cfg = getattr(cfg.data, table)
        df = ProcessingPipeline(table_cfg).fit_transform(df)

        id_cols = [c for c in df.columns if c.startswith("SK_ID")]
        feature_cols = [c for c in df.columns if c not in id_cols + ["TARGET"]]
        features_list.append(df.select(["SK_ID_CURR"] + feature_cols))

    logger.info(f"Joining {len(features_list)} tables with labels...")

    combined = labels.lazy()
    for i, df in enumerate(features_list):
        combined = combined.join(df.lazy(), on="SK_ID_CURR", how="left", suffix=f"_{i}")

    combined = combined.collect()

    logger.info(f"Combined: {combined.height} rows, {combined.width} cols")

    target_col = cfg.data.target.column
    id_col = cfg.data.target.id_column

    feature_cols = [c for c in combined.columns if c not in [target_col, id_col]]

    X_full = combined.select(feature_cols).to_pandas().values
    y_full = combined.select(target_col).to_numpy().ravel()

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Total samples: {X_full.shape[0]}")

    X_train, X_test, y_train, y_test = splitter.split_train_test(X_full, y_full)

    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    model_params = cfg.model.params.copy()

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

        final_model = model_factory(**model_params).build_model_pipeline()
        final_model.fit(X_train_best, y_train)
        mlflow.sklearn.log_model(final_model, "model")

    logger.info("Pipeline complete")

    if mlflow_active:
        mlflow.end_run()


if __name__ == "__main__":
    main()
