#!/usr/bin/env python
"""Generate credit risk predictions for submission.

Loads the best InferencePipeline from MLflow, processes all configured
tables (using application_test instead of application_train), and writes
a submission CSV.

Usage:
    RUN_MODE=prod uv run python scripts/predict.py
    RUN_MODE=dev  uv run python scripts/predict.py --output dev_submission.csv
    uv run python scripts/predict.py   # defaults to prod
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mlflow
import polars as pl
from credit_risk_models import InferencePipeline

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader

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


def load_inference_pipeline(run_mode: str) -> InferencePipeline:
    """Load the best InferencePipeline from MLflow.

    Falls back through modes: prod → dev → debug.

    Args:
        run_mode: Target run mode (prod / dev / debug).

    Returns:
        Deserialized InferencePipeline.

    Raises:
        RuntimeError: If no final_train run with a saved pipeline is found.
    """
    fallback_order = {
        "prod": ["prod", "dev", "debug"],
        "dev": ["dev", "debug"],
        "debug": ["debug"],
    }
    modes_to_try = fallback_order.get(run_mode, [run_mode])
    client = mlflow.MlflowClient()

    for mode in modes_to_try:
        experiment_name = f"final_train_{mode}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.test_roc_auc > 0",
            order_by=["metrics.test_roc_auc DESC"],
            max_results=1,
        )
        if runs.empty:
            continue

        run_id = runs.iloc[0]["run_id"]
        test_roc_auc = runs.iloc[0]["metrics.test_roc_auc"]

        if mode != run_mode:
            logger.warning(
                "No final_train_%s run found — falling back to final_train_%s",
                run_mode,
                mode,
            )
        logger.info(
            "Loading inference pipeline from run %s (test_roc_auc=%.4f, mode=%s)",
            run_id[:8],
            test_roc_auc,
            mode,
        )

        artifacts = [a.path for a in client.list_artifacts(run_id)]
        source = "MLflow artifact"
        path: Path | None = None

        if "inference_pipeline.pkl" in artifacts:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="inference_pipeline.pkl"
            )
            path = Path(local)
        elif f"inference_pipeline_{mode}.pkl" in artifacts:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"inference_pipeline_{mode}.pkl"
            )
            path = Path(local)
        else:
            from credit_risk.config.config import PROJECT_ROOT

            pkl_path = PROJECT_ROOT / "models" / f"inference_pipeline_{mode}.pkl"
            if pkl_path.exists():
                path = pkl_path
                source = "local file"

        if path is None:
            logger.warning("No InferencePipeline found for run %s", run_id[:8])
            continue

        logger.info("Loading InferencePipeline from %s: %s", source, path)
        return InferencePipeline.load(path)

    raise RuntimeError(
        f"No usable final_train run found for modes {modes_to_try}. "
        "Run scripts/final_train.py first."
    )


def main() -> None:
    """Entry point for generating credit risk predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict credit risk on application_test")
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path for the output submission CSV (default: predictions.csv)",
    )
    args = parser.parse_args()

    cfg = load_config()
    run_mode = cfg.run.mode
    logger.info("Prediction — mode: %s", run_mode)

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())

    inference = load_inference_pipeline(run_mode)
    logger.info(
        "Loaded: %d features, %d processing pipelines",
        len(inference.feature_names),
        len(inference.processing_pipelines),
    )

    loader = PLLazyDataLoader()
    included_tables = [t for t in ALL_TABLES if getattr(cfg.data, t).include]
    logger.info("Loading tables: %s", included_tables)

    raw_tables = {}
    for t in included_tables:
        is_test = t == "application"
        table_key = "application_test" if is_test else t
        raw_tables[t] = loader.load(table_key).collect()

    ids, probas = inference.predict(raw_tables)
    logger.info(
        "Predictions: %d rows, min=%.4f  mean=%.4f  max=%.4f",
        len(probas),
        probas.min(),
        probas.mean(),
        probas.max(),
    )

    id_column = cfg.data.target.id_column
    output_path = Path(args.output)
    submission = pl.DataFrame({id_column: ids, "TARGET": probas})
    submission.write_csv(output_path)
    logger.info("Submission saved: %s  (%d rows)", output_path, len(submission))


if __name__ == "__main__":
    main()
