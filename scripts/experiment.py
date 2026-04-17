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

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from credit_risk.config import load_config
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.features.aggregator import DataAggregator
from credit_risk.features.transformer import DataTransformer
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.cross_validator import CrossValidator, LGBMFactory
from credit_risk.models.metrics import ClassificationRankingMetrics

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(config=None):
    cfg = config or load_config("selection")
    ml_logger = MlflowLogger()

    parser = argparse.ArgumentParser(description="Run CV experiment on a single table")
    parser.add_argument("--table", type=str, default="application")
    args = parser.parse_args()

    table = args.table
    run_mode = cfg.run.mode
    logger.info(f"Running experiment on table: {table}, mode: {run_mode}")

    with ml_logger.start_run(run_name=f"{run_mode}_{table}"):
        ml_logger.log_flat_config(cfg)

        sample_frac = cfg.run.sample_fraction

        logger.info("Loading data...")
        loader = PLLazyDataLoader()
        labels = loader.load_labels()
        if sample_frac < 1.0:
            logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")
            labels = labels.collect().sample(fraction=sample_frac, seed=cfg.run.random_state).lazy()

        df = loader.load(table).join(labels, on="SK_ID_CURR", how="inner")
        df = df.collect()

        logger.info(f"{table}: {df.height} rows, {df.width} cols")

        cleaner = DataCleaner()
        imputer = DataImputer()
        aggregator = DataAggregator()
        transformer = DataTransformer()

        df = cleaner.clean(df, table, method=cfg.cleaner.method)
        df = imputer.impute(df, table, method=cfg.imputer.method)
        df = aggregator.aggregate(df.lazy(), table, method=cfg.aggregator.method).collect()
        df = transformer.transform(df, table=table, encoding=cfg.transformer.encoding)

        target_col = cfg.data.target.column
        id_col = cfg.data.target.id_column

        # Handle TARGET column renamed during join - extract before dropping
        y = None
        if target_col not in df.columns and "TARGET_right" in df.columns:
            y = df.select("TARGET_right").to_numpy().ravel()
            df = df.drop("TARGET_right")
        elif target_col in df.columns:
            y = df.select(target_col).to_numpy().ravel()
            df = df.drop(target_col)

        # Drop any suffix columns from joins
        suffix_cols = [c for c in df.columns if c.endswith("_right")]
        if suffix_cols:
            df = df.drop(suffix_cols)

        # Exclude string columns if encoding is none
        if cfg.transformer.encoding == "none":
            feature_cols = [
                c
                for c in df.columns
                if c not in [id_col, "SK_ID_BUREAU", "SK_ID_PREV", "TARGET_right"]
                and df.schema[c] != pl.String
            ]
        else:
            feature_cols = [
                c
                for c in df.columns
                if c not in [id_col, "SK_ID_BUREAU", "SK_ID_PREV", "TARGET_right"]
            ]
        logger.info(f"Using {len(feature_cols)} features")

        X = df.select(feature_cols).to_pandas()

        X = X.to_numpy(dtype=np.float64)

        logger.info(f"Data shape: {X.shape}, Features: {len(feature_cols)}")

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
        logger.info(
            f"  ROC AUC: {result.mean_scores['roc_auc']:.4f} ± {result.std_scores['roc_auc']:.4f}"
        )
        logger.info("=" * 60)

        for fold_idx, fold_score in enumerate(result.fold_scores):
            logger.info(f"Fold {fold_idx + 1}: ROC AUC = {fold_score['roc_auc']:.4f}")

        ml_logger.log_metrics(
            {
                "cv_roc_auc": result.mean_scores["roc_auc"],
                "cv_roc_auc_std": result.std_scores["roc_auc"],
                "n_folds": cfg.splitter.n_splits,
                "n_features": len(feature_cols),
            }
        )

        logger.info("Experiment complete")

        return result.mean_scores["roc_auc"]


if __name__ == "__main__":
    main()
