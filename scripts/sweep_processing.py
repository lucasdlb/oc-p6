#!/usr/bin/env python
"""Sweep processing methods using ConfigGrid.

Tests all combinations of cleaner, imputer, aggregator, and transformer methods.

Usage:
    RUN_MODE=debug uv run python scripts/sweep_processing.py
    uv run python scripts/sweep_processing.py
"""

from __future__ import annotations

import logging
import sys
import warnings

import mlflow

from credit_risk.config import ConfigGrid, load_config
from credit_risk.config.config import CONFIG_DIR
from credit_risk.mlflow_utils import MlflowLogger

sys.path.insert(0, "scripts")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    base_cfg = load_config("model")
    grid = ConfigGrid(CONFIG_DIR / "sweep.toml", base_config=base_cfg)
    logger.info(f"Running {len(grid)} configurations")

    mlflow.set_tracking_uri(base_cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment("sweep_processing")
    ml_logger = MlflowLogger()
    with ml_logger.start_run(run_name="sweep_parent2"):
        ml_logger.log_dict_artifact(base_cfg.model_dump(), "sweep_config.json")
        ml_logger.log_file_artifact(str(CONFIG_DIR / "sweep.toml"))
        ml_logger.log_grid_config(grid)
        logger.info("MLflow tracking enabled")

        from experiment import main as run_experiment

        results = []
        for i, cfg in enumerate(grid):
            logger.info(
                f"[{i + 1}/{len(grid)}] "
                f"cleaner={cfg.cleaner.method}, imputer={cfg.imputer.method}, "
                f"aggregator={cfg.aggregator.method}, "
                f"transformer={cfg.transformer.encoding}"
            )

            roc_auc = run_experiment(config=cfg)
            results.append((cfg, roc_auc))
            logger.info(f"  -> ROC AUC: {roc_auc:.4f}")

        logger.info("=" * 60)
        logger.info("All results:")
        for cfg, roc_auc in results:
            logger.info(
                f"  {cfg.cleaner.method}/{cfg.imputer.method}/{cfg.aggregator.method}/"
                f"{cfg.transformer.encoding}: ROC AUC = {roc_auc:.4f}"
            )


if __name__ == "__main__":
    main()
