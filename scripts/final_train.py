#!/usr/bin/env python
"""Final model training using features selected by rfe_cv and tuned hyperparameters.

Feature list is loaded from the best MLflow run in experiment "rfe_cv_all_{mode}".
Model hyperparameters are loaded from the best MLflow run in experiment "{mode}_tuning".

Usage:
    RUN_MODE=debug uv run python scripts/final_train.py
    RUN_MODE=dev   uv run python scripts/final_train.py
    uv run python scripts/final_train.py  # defaults to prod
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import shap
from matplotlib import pyplot as plt

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.data.transformation import TransformerRegistry
from credit_risk.interpret.shap_explainer import ShapExplainer
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.final_model import FinalModelTrainer
from credit_risk.models.model_factory import get_factory
from credit_risk.models.plotter import ModelPlotter
from credit_risk.models.resampler import create_resampler
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.threshold_selector import CVThresholdSelector
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.pipeline.table_transformer import TableTransformer

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
    "pos_cash_balance",
    "installments",
    "credit_card_balance",
]


# ---------------------------------------------------------------------------
# Feature list loading from MLflow
# ---------------------------------------------------------------------------


def load_features_from_mlflow(run_mode: str, tracking_uri: str) -> list[str]:
    """Load the best feature list from the rfe_cv_all MLflow experiment.

    Queries experiment "rfe_cv_all_{mode}", orders by cv_roc_auc descending,
    downloads "features.json" artifact from the top run.

    Args:
        run_mode: One of "debug", "dev", "prod".
        tracking_uri: MLflow tracking URI.

    Returns:
        List of selected feature names.

    Raises:
        RuntimeError: If no suitable run is found or artifact is missing.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"rfe_cv_all_{run_mode}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. "
            f"Run rfe_cv.py first (combined mode, no --table flag)."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.cv_roc_auc > 0",
        order_by=["metrics.cv_roc_auc DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            f"No completed runs with cv_roc_auc found in experiment '{experiment_name}'."
        )

    run_id = runs.iloc[0]["run_id"]
    logger.info(
        f"Loading features from run {run_id[:8]} "
        f"(cv_roc_auc={runs.iloc[0].get('metrics.cv_roc_auc', 'n/a'):.4f})"
    )

    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="features.json"
    )
    feature_data = json.loads(Path(artifact_path).read_text())
    features = feature_data["features"]
    logger.info(f"Loaded {len(features)} features from MLflow")
    return features


# ---------------------------------------------------------------------------
# Tuned hyperparameters loading from MLflow
# ---------------------------------------------------------------------------


def load_params_from_mlflow(run_mode: str, tracking_uri: str) -> tuple[dict, str] | None:
    """Load best hyperparameters from the tuning MLflow experiment.

    Returns (params_dict, model_name) or None if no suitable run found.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{run_mode}_tuning"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Tuning experiment '{experiment_name}' not found — using config params")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.best_roc_auc > 0",
        order_by=["metrics.best_roc_auc DESC"],
        max_results=10,
    )

    for _, run_row in runs.iterrows():
        run_id = run_row["run_id"]
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        params = dict(run.data.params)
        if "n_estimators" in params or "learning_rate" in params:
            best_roc_auc = run_row.get("metrics.best_roc_auc", 0)
            model_name = params.get("best_model", "lgbm")
            logger.info(
                f"Loaded tuned params from run {run_id[:8]} "
                f"(best_roc_auc={best_roc_auc:.4f}, model={model_name})"
            )
            return params, model_name

    logger.warning("No tuning run with hyperparameters found — using config params")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config("model", "resampling")
    run_mode = cfg.run.mode
    logger.info(f"Final training in mode: {run_mode}")

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment(f"final_train_{run_mode}")
    ml_logger = MlflowLogger()

    # ── Load feature list from MLflow ────────────────────────────────────────
    best_features = load_features_from_mlflow(run_mode, cfg.output.mlflow_tracking_uri())

    # ── Load hyperparameters (MLflow tuning run → fallback to config) ────────
    tuning_result = load_params_from_mlflow(run_mode, cfg.output.mlflow_tracking_uri())
    if tuning_result is not None:
        raw_params, best_model_name = tuning_result
        # MLflow stores all params as strings — cast numerics
        model_params: dict = {}
        for k, v in raw_params.items():
            if k in ("verbose", "run_mode", "models", "best_model", "x_transform", "nan_fill"):
                continue
            try:
                model_params[k] = int(v) if "." not in v else float(v)
            except (ValueError, TypeError):
                model_params[k] = v
    else:
        model_params = cfg.model.params.copy()
        best_model_name = cfg.model.model_type

    model_params["verbose"] = -1
    logger.info(f"Training with model={best_model_name}, params={model_params}")

    # ── Splitter + resampler ────────────────────────────────────────────────
    resampler = None
    if cfg.resampling and cfg.resampling.enabled:
        method = cfg.resampling.method
        kwargs: dict = {
            "sampling_strategy": cfg.resampling.sampling_strategy,
            "random_state": cfg.resampling.random_state,
        }
        if method == "smote":
            kwargs["k_neighbors"] = cfg.resampling.k_neighbors
        resampler = create_resampler(method, **kwargs)
        logger.info(f"Resampling: {method}, strategy={cfg.resampling.sampling_strategy}")

    splitter = TrainTestCVSplitter.from_config(cfg)
    if resampler is not None:
        splitter.set_resampler(resampler)

    # ── Load labels, lock away test set ─────────────────────────────────────
    loader = PLLazyDataLoader()
    labels_df = loader.load_labels().collect()

    if cfg.run.sample_fraction < 1.0:
        logger.info(f"Sampling {cfg.run.sample_fraction * 100:.0f}% of data")
        labels_df = labels_df.sample(fraction=cfg.run.sample_fraction, seed=cfg.run.random_state)

    ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
    y = labels_df.select(cfg.data.target.column).to_numpy().ravel()
    ids_train, ids_test, _, _ = splitter.split_train_test(ids, y)

    labels_train_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

    # ── Process all tables via TableTransformer (leak-free) ─────────────────
    included_tables = [t for t in TABLES if getattr(cfg.data, t).include]
    logger.info(f"Processing tables: {included_tables}")

    raw_tables = {}
    for table in included_tables:
        logger.info(f"  Loading {table}...")
        raw_tables[table] = loader.load(table).collect()

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

    X_train_all, X_test_all, y_train, y_test, all_feature_names = table_transformer.fit_transform(
        tables=raw_tables,
        labels=labels_train_df,
        train_ids=set(ids_train),
        val_ids=set(ids_test),
    )

    logger.info(
        f"Processed: {X_train_all.shape[1]} total features, "
        f"{X_train_all.shape[0]} train, {X_test_all.shape[0]} test samples"
    )

    # ── Filter to selected features ─────────────────────────────────────────
    available = set(all_feature_names)
    feature_cols = [f for f in best_features if f in available]
    missing = [f for f in best_features if f not in available]
    if missing:
        logger.warning(f"{len(missing)} selected features not found after processing: {missing}")
    logger.info(f"Using {len(feature_cols)} selected features")

    feat_idx = [all_feature_names.index(f) for f in feature_cols]
    X_train = X_train_all[:, feat_idx]
    X_test = X_test_all[:, feat_idx]

    # ── Train and evaluate ───────────────────────────────────────────────────
    model_factory = get_factory(
        best_model_name,
        cfg.model.x_transform,
        cfg.model.nan_fill,
    )
    threshold_selector = CVThresholdSelector(
        splitter=splitter,
        model_factory=model_factory,
        direction="maximize",
        metric="f1",
    )
    trainer = FinalModelTrainer(model_factory, threshold_selector=threshold_selector)
    result = trainer.train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_cols, model_params
    )

    logger.info(f"Test ROC AUC:  {result.test_roc_auc:.4f}")
    logger.info(f"Test F1:       {result.test_f1:.4f}")
    logger.info(f"Test Recall:   {result.test_recall:.4f}")
    logger.info(f"Test Precision:{result.test_precision:.4f}")
    logger.info(f"Threshold:     {result.optimal_threshold:.2f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    final_model = model_factory(**model_params).build_model_pipeline()
    final_model.fit(X_train, y_train)
    y_pred_proba = final_model.predict_proba(X_test)

    plots_path = cfg.output.models_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    plotter = ModelPlotter()
    plotter.plot_all(
        y_test, y_pred_proba, threshold=result.optimal_threshold, output_dir=plots_path
    )

    # ── Save model + feature list ────────────────────────────────────────────
    model_dir = Path(cfg.output.models_path)
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"final_model_{run_mode}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Model saved: {model_path}")

    feature_path = model_dir / f"features_{run_mode}.json"
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Features saved: {feature_path}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    logger.info("Computing SHAP feature importance...")
    numpy_X = X_train if isinstance(X_train, np.ndarray) else X_train.numpy()
    shap_exp = ShapExplainer()
    shap_exp.fit(final_model, numpy_X)
    shap_vals, _ = shap_exp.global_importance(numpy_X, n_samples=300)

    shap_plot_path = plots_path / "shap_summary.png"
    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, numpy_X[:300], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP plot saved: {shap_plot_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with ml_logger.start_run(run_name=f"{run_mode}_final_model"):
        ml_logger.log_flat_config(cfg)
        ml_logger.log_params({"model_type": best_model_name, "n_features": len(feature_cols)})
        ml_logger.log_params(model_params)
        ml_logger.log_metrics(
            {
                "test_roc_auc": result.test_roc_auc,
                "test_f1": result.test_f1,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "optimal_threshold": result.optimal_threshold,
            }
        )
        ml_logger.log_model(final_model, "final_model")
        ml_logger.log_file_artifact(str(model_path))
        ml_logger.log_file_artifact(str(feature_path))
        ml_logger.log_file_artifact(str(shap_plot_path))
        plotter.log_to_mlflow(True)

    logger.info("Final training complete")


if __name__ == "__main__":
    main()
