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

import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.cross_validator import CVMetrics
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


def main(config=None):
    cfg = config or load_config("selection", "model")
    ml_logger = MlflowLogger()

    parser = argparse.ArgumentParser(description="Run CV experiment on a single table")
    parser.add_argument("--table", type=str, default="application")
    args = parser.parse_args()

    table = args.table
    run_mode = cfg.run.mode
    logger.info(f"Running experiment on table: {table}, mode: {run_mode}")

    with ml_logger.start_run(run_name=f"{run_mode}_{table}"):
        ml_logger.log_flat_config(cfg)

        splitter = TrainTestCVSplitter.from_config(cfg=cfg)

        sample_frac = cfg.run.sample_fraction

        loader = PLLazyDataLoader()
        labels = loader.load_labels()

        labels_df = labels.collect()
        ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
        y = labels_df.select(cfg.data.target.column).to_numpy().ravel()

        ids_train, _, _, _ = splitter.split_train_test(ids, y)
        labels_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

        if sample_frac < 1.0:
            logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")
            labels_df = labels_df.sample(fraction=sample_frac, seed=cfg.run.random_state)

        logger.info("Loading data...")
        table_raw = loader.load(table).collect()

        tables = {table: table_raw}

        pipeline_factories = {
            t: lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build() for t in tables
        }
        table_transformer = TableTransformer(pipeline_factories=pipeline_factories)
        cv = ProcessingCV(
            table_transformer=table_transformer,
            splitter=splitter,
            model_factory=get_factory(cfg.model.model_type, cfg.model.x_transform),
        )

        logger.info(f"Running {cfg.splitter.n_splits}-fold cross-validation...")

        result = cv.validate(
            tables=tables,
            labels=labels_df,
            model_params=cfg.model.params,
        )

        scores = CVMetrics.compute(result, ClassificationRankingMetrics())

        n_features = result.n_features
        logger.info("=" * 60)
        logger.info(f"CV Results ({cfg.splitter.n_splits} folds, {n_features} features):")
        auc_mean = scores.mean_scores["roc_auc"]
        auc_std = scores.std_scores["roc_auc"]
        logger.info(f"  ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}")

        for fold_idx, fold_score in enumerate(scores.fold_scores):
            logger.info(f"Fold {fold_idx + 1}: ROC AUC = {fold_score['roc_auc']:.4f}")

        ml_logger.log_metrics(
            {
                "cv_roc_auc": auc_mean,
                "cv_roc_auc_std": auc_std,
                "n_folds": cfg.splitter.n_splits,
                "n_features": n_features,
            }
        )

        logger.info("Experiment complete")

        return auc_mean


if __name__ == "__main__":
    main()
