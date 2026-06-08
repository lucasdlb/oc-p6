#!/usr/bin/env python
"""RFE-CV feature selection on the joined feature matrix of given tables.

Zero-leakage pipeline:
  1. Split labels into train / test — test is locked away and never touched here.
  2. Load raw tables (train IDs only).
  3. TableTransformer fits processing pipelines on train IDs only.
  4. BackwardFeatureSelector runs entirely on X_train using internal CV folds.
  5. Feature list + CV metrics logged as a MLflow artifact ("features.json").

Test-set evaluation happens only in final_train.py after sweep + rfe + tuning.

Passing a single table in --tables is equivalent to per-table isolation.
Passing multiple tables runs RFE on their joined feature matrix.
Omitting --tables uses all included tables from config.

Usage
-----
    # All included tables (default)
    RUN_MODE=debug uv run python scripts/rfe_cv.py

    # Single table
    RUN_MODE=debug uv run python scripts/rfe_cv.py --tables application

    # Subset of tables
    RUN_MODE=debug uv run python scripts/rfe_cv.py --tables application bureau
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

import mlflow
import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk_processing.data.transformation import TransformerRegistry
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.cross_validator import CVMetrics
from credit_risk.models.feature_selector import BackwardFeatureSelector
from credit_risk.models.importance import get_importance_class
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.pipeline.cv_pipeline import ProcessingCV
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.pipeline.table_transformer import TableTransformer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RFE-CV feature selection")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        metavar="TABLE",
        help="Tables to include. Defaults to all included tables from config.",
    )
    args = parser.parse_args()

    cfg = load_config("selection", "importance", "model")
    run_mode = cfg.run.mode

    # ── Resolve table list ───────────────────────────────────────────────────
    included = [t for t in ALL_TABLES if getattr(cfg.data, t).include]
    tables = args.tables if args.tables else included

    unknown = [t for t in tables if t not in ALL_TABLES]
    if unknown:
        raise ValueError(f"Unknown tables: {unknown}. Available: {ALL_TABLES}")

    logger.info(f"Mode: {run_mode}, tables: {tables}")

    # ── Shared objects ───────────────────────────────────────────────────────
    splitter = TrainTestCVSplitter.from_config(cfg)
    model_factory = get_factory(
        cfg.model.model_type,
        cfg.model.x_transform,
        cfg.model.nan_fill,
    )
    importance_strategy = get_importance_class(cfg.importance.method)()
    model_params = cfg.model.params.copy()

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
    ml_logger = MlflowLogger()

    # ── Load labels, isolate train IDs — test set locked away ───────────────
    loader = PLLazyDataLoader()
    labels_df = loader.load_labels().collect()

    ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
    y = labels_df.select(cfg.data.target.column).to_numpy().ravel()

    ids_train, _, _, _ = splitter.split_train_test(ids, y)

    labels_train_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

    if cfg.run.sample_fraction < 1.0:
        logger.info(f"Sampling {cfg.run.sample_fraction * 100:.0f}% of training data")
        labels_train_df = labels_train_df.sample(
            fraction=cfg.run.sample_fraction,
            seed=cfg.run.random_state,
        )
        ids_train = labels_train_df.select(cfg.data.target.id_column).to_numpy().ravel()

    logger.info(f"Train samples: {len(ids_train)}")

    # ── Load raw tables ──────────────────────────────────────────────────────
    logger.info(f"Loading tables: {tables}")
    raw_tables = {t: loader.load(t).collect() for t in tables}

    # ── Build TableTransformer + ProcessingCV ────────────────────────────────
    pipeline_factories = {
        t: (lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build()) for t in tables
    }
    cross_transformer = TransformerRegistry.get(cfg.data.cross.transformer)()

    table_transformer = TableTransformer(
        pipeline_factories=pipeline_factories,
        id_column=cfg.data.target.id_column,
        target_column=cfg.data.target.column,
        cross_transformer=cross_transformer,
    )

    processing_cv = ProcessingCV(
        table_transformer=table_transformer,
        splitter=splitter,
        model_factory=model_factory,
        importance_strategy=importance_strategy,
    )

    # ── Backward feature selection (train only, internal CV folds) ───────────
    selector = BackwardFeatureSelector(
        processing_cv=processing_cv,
        metrics=ClassificationRankingMetrics(),
        selection_metric_name="roc_auc",
        min_features=cfg.selection.min_features,
        tolerance=cfg.selection.tolerance,
        nb_remove_features=cfg.selection.nb_remove_features,
        verbose=True,
    )
    best_features, best_cv_result = selector.eliminate(raw_tables, labels_train_df, model_params)

    scores = CVMetrics.compute(best_cv_result, ClassificationRankingMetrics())
    cv_roc_auc = scores.mean_scores.get("roc_auc", 0.0)
    cv_roc_auc_std = scores.std_scores.get("roc_auc", 0.0)

    logger.info(f"Selected {len(best_features)} features")
    logger.info(f"CV ROC AUC: {cv_roc_auc:.4f} ± {cv_roc_auc_std:.4f}")

    # ── Log to MLflow ────────────────────────────────────────────────────────
    table_tag = tables[0] if len(tables) == 1 else "all"
    experiment_name = f"rfe_cv_{table_tag}_{run_mode}"
    mlflow.set_experiment(experiment_name)

    feature_data = {
        "features": best_features,
        "n_features": len(best_features),
        "tables": tables,
        "cv_roc_auc": cv_roc_auc,
        "cv_roc_auc_std": cv_roc_auc_std,
        "mode": run_mode,
    }

    with ml_logger.start_run(run_name=f"{run_mode}_{table_tag}"):
        ml_logger.log_flat_config(cfg)
        ml_logger.log_params(
            {
                "run_mode": run_mode,
                "tables": json.dumps(tables),
                "n_tables": len(tables),
            }
        )
        ml_logger.log_metrics(
            {
                "cv_roc_auc": cv_roc_auc,
                "cv_roc_auc_std": cv_roc_auc_std,
                "n_features": len(best_features),
            }
        )
        mlflow.set_tags({"mode": run_mode, "tables": json.dumps(tables)})
        ml_logger.log_dict_artifact(feature_data, "features.json", artifact_path="")

    logger.info("RFE-CV complete")
    logger.info(f"  Features:   {len(best_features)}")
    logger.info(f"  CV ROC AUC: {cv_roc_auc:.4f} ± {cv_roc_auc_std:.4f}")


if __name__ == "__main__":
    main()
