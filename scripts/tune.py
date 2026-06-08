#!/usr/bin/env python
"""Optuna hyperparameter tuning across all tables with zero-leakage ProcessingCV.

Zero-leakage pipeline — same structure as rfe_cv.py:
  1. Split labels into train / test — test is locked away immediately.
  2. Load raw tables (train IDs only).
  3. TableTransformer fits processing pipelines on train IDs only per fold.
  4. Feature list is loaded from the best rfe_cv_all_{mode} MLflow run and
     applied as a feature_mask — tuning is performed on the same feature
     subset that will be used in final_train.py.

Usage:
    RUN_MODE=debug uv run python scripts/tune.py
    RUN_MODE=dev   uv run python scripts/tune.py
    uv run python scripts/tune.py  # defaults to prod
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

import mlflow
import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk_processing.data.transformation import TransformerRegistry
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.processing_tuner import ProcessingTuner
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.pipeline.table_transformer import TableTransformer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_TABLES = [
    "application",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments",
    "credit_card_balance",
]


def _load_feature_mask(run_mode: str, tracking_uri: str) -> list[str] | None:
    """Load selected features from the best rfe_cv_all_{mode} MLflow run.

    Returns None if no suitable run is found — tuning will then run on all
    features (degraded mode, with a warning).
    """
    experiment_name = "rfe_cv_all_dev"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(
            "No MLflow experiment '%s' found. Run rfe_cv.py first. "
            "Tuning will run on all features.",
            experiment_name,
        )
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.cv_roc_auc > 0",
        order_by=["metrics.cv_roc_auc DESC"],
        max_results=1,
    )
    if runs.empty:
        logger.warning(
            "No completed runs in '%s'. Tuning will run on all features.",
            experiment_name,
        )
        return None

    run_id = runs.iloc[0]["run_id"]
    cv_roc_auc = runs.iloc[0]["metrics.cv_roc_auc"]
    logger.info(
        "Loading feature mask from run %s (cv_roc_auc=%.4f)",
        run_id[:8],
        cv_roc_auc,
    )

    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="features.json"
    )
    data = json.loads(pathlib.Path(artifact_path).read_text())
    features = data["features"]
    logger.info("Feature mask: %d features loaded", len(features))
    return features


def main(config=None):
    cfg = config or load_config("tuning", "model")
    run_mode = cfg.run.mode

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment(f"{run_mode}_tuning")
    ml_logger = MlflowLogger()

    # ── Load feature mask from rfe_cv ────────────────────────────────────────
    feature_mask = _load_feature_mask(run_mode, cfg.output.mlflow_tracking_uri())

    # ── Splitter ─────────────────────────────────────────────────────────────
    splitter = TrainTestCVSplitter.from_config(cfg=cfg)

    # ── Load labels, isolate train IDs — test set locked away ────────────────
    loader = PLLazyDataLoader()
    labels_df = loader.load_labels().collect()

    ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
    y_full = labels_df.select(cfg.data.target.column).to_numpy().ravel()
    ids_train, _, _, _ = splitter.split_train_test(ids, y_full)

    labels_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

    if cfg.run.sample_fraction < 1.0:
        labels_df = labels_df.sample(
            fraction=cfg.run.sample_fraction,
            seed=cfg.run.random_state,
        )

    logger.info("Train samples: %d", len(labels_df))

    # ── Load raw tables ───────────────────────────────────────────────────────
    included_tables = [t for t in ALL_TABLES if getattr(cfg.data, t).include]
    logger.info("Tables: %s", included_tables)

    raw_tables = {}
    for table in included_tables:
        logger.info("Loading %s...", table)
        raw_tables[table] = loader.load(table).collect()

    # ── Build TableTransformer ────────────────────────────────────────────────
    pipeline_factories = {
        t: (lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build())
        for t in included_tables
    }
    cross_transformer = TransformerRegistry.get(cfg.data.cross.transformer)()

    table_transformer = TableTransformer(
        pipeline_factories=pipeline_factories,
        id_column=cfg.data.target.id_column,
        target_column=cfg.data.target.column,
        cross_transformer=cross_transformer,
    )

    # ── Run Optuna tuning ─────────────────────────────────────────────────────
    with ml_logger.start_run(run_name=f"{run_mode}_tuning"):
        ml_logger.log_flat_config(cfg)
        ml_logger.log_params(
            {
                "run_mode": run_mode,
                "n_trials": cfg.tuning.n_trials,
                "models": ",".join(cfg.tuning.models),
                "n_features_mask": len(feature_mask) if feature_mask else "all",
            }
        )

        tuner = ProcessingTuner(
            table_transformer=table_transformer,
            splitter=splitter,
            tuning_config=cfg.tuning,
            metrics=ClassificationRankingMetrics(),
        )

        results = tuner.optimize(
            raw_tables,
            labels_df,
            feature_mask=feature_mask,
        )

        best_name = max(results, key=lambda k: results[k]["best_value"])
        best = results[best_name]

        # Log overall best on the parent run
        ml_logger.log_params({"best_model": best_name})
        ml_logger.log_metric("best_roc_auc", best["best_value"])
        for k, v in best.get("best_scores", {}).items():
            ml_logger.log_metric(f"best_{k}", v)

        # Save overall best config as artifact
        ml_logger.log_dict_artifact(
            {
                "best_model_type": best_name,
                "best_params": best["best_params"],
                "best_metrics": best.get("best_scores", {}),
                "x_transform": cfg.tuning.x_transform,
                "nan_fill": cfg.tuning.nan_fill,
                "n_features_mask": len(feature_mask) if feature_mask else None,
                "run_mode": run_mode,
            },
            "best_config.json",
            artifact_path="",
        )

        logger.info("=" * 60)
        logger.info("Results:")
        for name, result in results.items():
            logger.info("  %s: ROC AUC = %.4f", name, result["best_value"])
        logger.info("Best overall: %s (%.4f)", best_name, best["best_value"])
        logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
