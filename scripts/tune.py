#!/usr/bin/env python
"""Optuna hyperparameter tuning across all tables with zero-leakage ProcessingCV.

Each Optuna trial runs a full ProcessingCV.validate() call, which performs
per-fold fit/transform of all tables via TableTransformer.

Usage:
    RUN_MODE=debug uv run python scripts/tune.py
    RUN_MODE=dev   uv run python scripts/tune.py
    uv run python scripts/tune.py  # defaults to prod
"""

from __future__ import annotations

import logging
import warnings

import mlflow
import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.data.transformation import TransformerRegistry
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


def main(config=None):
    cfg = config or load_config("tuning", "model")
    run_mode = cfg.run.mode

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment(f"{run_mode}_tuning")
    ml_logger = MlflowLogger()

    with ml_logger.start_run(run_name=f"{run_mode}_tuning"):
        ml_logger.log_flat_config(cfg)
        ml_logger.log_params(
            {
                "run_mode": run_mode,
                "n_trials": cfg.tuning.n_trials,
                "models": ",".join(cfg.tuning.models),
            }
        )

        splitter = TrainTestCVSplitter.from_config(cfg=cfg)

        loader = PLLazyDataLoader()
        labels = loader.load_labels().collect()

        ids = labels.select(cfg.data.target.id_column).to_numpy().ravel()
        y_full = labels.select(cfg.data.target.column).to_numpy().ravel()
        ids_train, _, _, _ = splitter.split_train_test(ids, y_full)

        labels_df = labels.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

        if cfg.run.sample_fraction < 1.0:
            labels_df = labels_df.sample(
                fraction=cfg.run.sample_fraction,
                seed=cfg.run.random_state,
            )

        included_tables = [name for name in ALL_TABLES if getattr(cfg.data, name).include]
        logger.info(f"Tables: {included_tables}")

        raw_tables = {}
        for table in included_tables:
            logger.info(f"Loading {table}...")
            raw_tables[table] = loader.load(table).collect()

        pipeline_factories = {
            t: lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build()
            for t in included_tables
        }
        cross_transformer = TransformerRegistry.get(cfg.data.cross.transformer)()

        table_transformer = TableTransformer(
            pipeline_factories=pipeline_factories,
            id_column=cfg.data.target.id_column,
            target_column=cfg.data.target.column,
            cross_transformer=cross_transformer,
        )

        tuner = ProcessingTuner(
            table_transformer=table_transformer,
            splitter=splitter,
            tuning_config=cfg.tuning,
            metrics=ClassificationRankingMetrics(),
            mlflow_logging=False,
        )

        results = tuner.optimize(raw_tables, labels_df)

        best_name = max(results, key=lambda k: results[k]["best_value"])
        ml_logger.log_params({"best_model": best_name})
        ml_logger.log_metric("best_roc_auc", results[best_name]["best_value"])

        logger.info("=" * 60)
        logger.info("Results:")
        for name, result in results.items():
            logger.info(f"  {name}: ROC AUC = {result['best_value']:.4f}")
        logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
