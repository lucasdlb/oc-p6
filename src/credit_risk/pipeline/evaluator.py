"""Zero-leakage CV evaluator — runs ProcessingCV on a set of tables.

This module is the programmatic entry point for cross-validated evaluation
of processing configurations.  It is used by sweep_processing.py to score
each processing combo, and can be called from any orchestration script.

It is intentionally not a CLI script — use sweep_processing.py to drive
grid searches or rfe_cv.py for feature selection.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import polars as pl

from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.data.transformation import TransformerRegistry
from credit_risk.models.cross_validator import CVMetrics
from credit_risk.models.metrics import ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.pipeline.cv_pipeline import ProcessingCV
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.pipeline.table_transformer import TableTransformer

if TYPE_CHECKING:
    from credit_risk.config.models import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)


def run_cv(
    cfg: Config,
    tables: list[str],
) -> tuple[float, float, int]:
    """Run zero-leakage cross-validation for the given tables and config.

    Locks the test set away immediately (only train IDs are used).
    Processes each table via a fresh pipeline per fold — no leakage.

    Args:
        cfg: Fully resolved Config (model + data settings).
        tables: List of table names to include in the feature matrix.

    Returns:
        Tuple of (cv_roc_auc_mean, cv_roc_auc_std, n_features).
    """
    run_mode = cfg.run.mode
    logger.info(f"CV evaluation — tables: {tables}, mode: {run_mode}")

    splitter = TrainTestCVSplitter.from_config(cfg=cfg)

    loader = PLLazyDataLoader()
    labels_df = loader.load_labels().collect()

    ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
    y = labels_df.select(cfg.data.target.column).to_numpy().ravel()

    # Lock test set away — CV operates on train IDs only.
    ids_train, _, _, _ = splitter.split_train_test(ids, y)
    labels_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

    if cfg.run.sample_fraction < 1.0:
        logger.info(f"Sampling {cfg.run.sample_fraction * 100:.0f}% of training data")
        labels_df = labels_df.sample(
            fraction=cfg.run.sample_fraction,
            seed=cfg.run.random_state,
        )

    logger.info("Loading raw tables...")
    raw_tables = {t: loader.load(t).collect() for t in tables}

    pipeline_factories = {
        t: (lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build()) for t in raw_tables
    }

    cross_transformer = TransformerRegistry.get(cfg.data.cross.transformer)()

    table_transformer = TableTransformer(
        pipeline_factories=pipeline_factories,
        id_column=cfg.data.target.id_column,
        target_column=cfg.data.target.column,
        cross_transformer=cross_transformer,
    )

    cv = ProcessingCV(
        table_transformer=table_transformer,
        splitter=splitter,
        model_factory=get_factory(
            cfg.model.model_type,
            cfg.model.x_transform,
            cfg.model.nan_fill,
        ),
    )

    logger.info(f"Running {cfg.splitter.n_splits}-fold CV...")

    result = cv.validate(
        tables=raw_tables,
        labels=labels_df,
        model_params=cfg.model.params,
    )

    scores = CVMetrics.compute(result, ClassificationRankingMetrics())
    auc_mean = scores.mean_scores["roc_auc"]
    auc_std = scores.std_scores["roc_auc"]
    n_features = result.n_features

    logger.info("=" * 60)
    logger.info(f"CV Results ({cfg.splitter.n_splits} folds, {n_features} features):")
    logger.info(f"  ROC AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    for fold_idx, fold_score in enumerate(scores.fold_scores):
        logger.info(f"  Fold {fold_idx + 1}: ROC AUC = {fold_score['roc_auc']:.4f}")

    return auc_mean, auc_std, n_features
