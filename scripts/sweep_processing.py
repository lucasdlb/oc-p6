#!/usr/bin/env python
"""Sweep processing configurations using ConfigGrid.

Tests all combinations defined in sweep_processing.toml.
Each axis in the TOML becomes a sweep dimension (cartesian product).

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
    base_cfg = load_config("selection", "model")
    grid = ConfigGrid(CONFIG_DIR / "sweep_processing.toml", base_config=base_cfg)
    logger.info(f"Running {len(grid)} configurations")
    logger.info(f"Grid axes: {list(grid.axes.keys())}")

    mlflow.set_tracking_uri(base_cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment("sweep_processing")
    ml_logger = MlflowLogger()
    with ml_logger.start_run(run_name="sweep_processing"):
        ml_logger.log_dict_artifact(base_cfg.model_dump(), "base_config.json")
        ml_logger.log_file_artifact(str(CONFIG_DIR / "sweep_processing.toml"))
        ml_logger.log_grid_config(grid)
        logger.info("MLflow tracking enabled")

        from experiment import main as run_experiment

        results = []
        for i, cfg in enumerate(grid):
            axis_desc = _format_cfg(cfg, grid.axes)
            logger.info(f"[{i + 1}/{len(grid)}] {axis_desc}")

            roc_auc = run_experiment(config=cfg)
            results.append((axis_desc, roc_auc))
            logger.info(f"  -> ROC AUC: {roc_auc:.4f}")

        logger.info("=" * 60)
        logger.info("All results:")
        for axis_desc, roc_auc in results:
            logger.info(f"  {axis_desc}: ROC AUC = {roc_auc:.4f}")


def _format_cfg(cfg, axes: dict[str, list]) -> str:
    """Format config as readable sweep axis values."""
    parts = []
    for dotted_key in axes.keys():
        keys = dotted_key.split(".")
        val = cfg
        for k in keys:
            val = getattr(val, k)
        parts.append(f"{dotted_key}={val}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
