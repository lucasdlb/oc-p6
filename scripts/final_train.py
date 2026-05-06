#!/usr/bin/env python
"""Final model training — loads features from rfe_cv and best params from tuning.

Zero-leakage pipeline — same structure as rfe_cv.py:
  1. Split labels into train / test — test locked away until evaluation.
  2. Process all tables via TableTransformer (fit on train, transform both).
  3. Filter to selected features from rfe_cv MLflow artifact.
  4. Train final model with best hyperparameters from tuning MLflow artifact.
  5. Evaluate on held-out test set, compute SHAP, save model + plots.

Usage:
    RUN_MODE=debug uv run python scripts/final_train.py
    RUN_MODE=dev   uv run python scripts/final_train.py
    uv run python scripts/final_train.py   # defaults to prod
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
from credit_risk.mlflow_utils import MlflowLogger
from credit_risk.models.final_model import FinalModelTrainer
from credit_risk.models.plotter import ModelPlotter
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.threshold_selector import SimpleThresholdSelector
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


# ---------------------------------------------------------------------------
# MLflow loaders
# ---------------------------------------------------------------------------


def load_features(run_mode: str) -> list[str]:
    """Load selected features from best rfe_cv_all MLflow run.

    Falls back through modes: prod → dev → debug.
    """
    fallback_order = {"prod": ["prod", "dev", "debug"], "dev": ["dev", "debug"], "debug": ["debug"]}
    modes_to_try = fallback_order.get(run_mode, [run_mode])

    for mode in modes_to_try:
        experiment_name = f"rfe_cv_all_{mode}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.cv_roc_auc > 0",
            order_by=["metrics.cv_roc_auc DESC"],
            max_results=1,
        )
        if runs.empty:
            continue

        run_id = runs.iloc[0]["run_id"]
        cv_roc_auc = runs.iloc[0]["metrics.cv_roc_auc"]
        if mode != run_mode:
            logger.warning(
                "No rfe_cv_all_%s run found — falling back to rfe_cv_all_%s", run_mode, mode
            )
        logger.info("Features from run %s (cv_roc_auc=%.4f, mode=%s)", run_id[:8], cv_roc_auc, mode)

        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="features.json")
        data = json.loads(Path(path).read_text())
        logger.info("Loaded %d features", data["n_features"])
        return data["features"]

    raise RuntimeError(f"No rfe_cv_all run found for any of {modes_to_try}. Run rfe_cv.py first.")


def load_best_config(run_mode: str) -> tuple[str, dict] | None:
    """Load best model type and params from tuning MLflow experiment.

    Falls back through modes: prod → dev → debug.
    Returns (model_name, params_dict) or None if no tuning run found.
    """
    fallback_order = {"prod": ["prod", "dev", "debug"], "dev": ["dev", "debug"], "debug": ["debug"]}
    modes_to_try = fallback_order.get(run_mode, [run_mode])

    client = mlflow.MlflowClient()

    for mode in modes_to_try:
        experiment_name = f"{mode}_tuning"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.best_roc_auc > 0",
            order_by=["metrics.best_roc_auc DESC"],
            max_results=5,
        )
        if runs.empty:
            continue

        for _, row in runs.iterrows():
            run = client.get_run(row["run_id"])
            best_model = run.data.params.get("best_model")
            if best_model is None:
                continue

            best_roc_auc = row.get("metrics.best_roc_auc", 0.0)
            if mode != run_mode:
                logger.warning("No %s_tuning run found — falling back to %s_tuning", run_mode, mode)
            logger.info(
                "Tuning run %s: best_model=%s  best_roc_auc=%.4f (mode=%s)",
                row["run_id"][:8],
                best_model,
                best_roc_auc,
                mode,
            )

            child_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_type = '{best_model}'",
            )
            for child in child_runs:
                artifacts = [a.path for a in client.list_artifacts(child.info.run_id)]
                if "best_config.json" in artifacts:
                    path = mlflow.artifacts.download_artifacts(
                        run_id=child.info.run_id, artifact_path="best_config.json"
                    )
                    config = json.loads(Path(path).read_text())
                    params = config.get("best_params", {})
                    logger.info(
                        "Loaded best_config.json from %s run %s (%d params)",
                        best_model,
                        child.info.run_id[:8],
                        len(params),
                    )
                    return best_model, params

    logger.warning("No best_config.json found in any tuning run — will use config params.")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config("model")
    run_mode = cfg.run.mode
    logger.info("Final training — mode: %s", run_mode)

    mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri())
    mlflow.set_experiment(f"final_train_{run_mode}")
    ml_logger = MlflowLogger()

    # ── Load features and hyperparameters from MLflow ────────────────────────
    best_features = load_features(run_mode)

    tuning_result = load_best_config(run_mode)
    if tuning_result is not None:
        best_model_name, model_params = tuning_result
    else:
        best_model_name = cfg.model.model_type
        model_params = cfg.model.params.copy()

    logger.info("Model: %s  params: %s", best_model_name, model_params)

    # ── Splitter ─────────────────────────────────────────────────────────────
    splitter = TrainTestCVSplitter.from_config(cfg)

    # ── Load labels, lock away test set ─────────────────────────────────────
    loader = PLLazyDataLoader()
    labels_df = loader.load_labels().collect()

    ids = labels_df.select(cfg.data.target.id_column).to_numpy().ravel()
    y = labels_df.select(cfg.data.target.column).to_numpy().ravel()
    ids_train, ids_test, _, _ = splitter.split_train_test(ids, y)

    labels_train_df = labels_df.filter(pl.col(cfg.data.target.id_column).is_in(ids_train))

    if cfg.run.sample_fraction < 1.0:
        logger.info("Sampling %.0f%% of training data", cfg.run.sample_fraction * 100)
        labels_train_df = labels_train_df.sample(
            fraction=cfg.run.sample_fraction, seed=cfg.run.random_state
        )
        ids_train = labels_train_df.select(cfg.data.target.id_column).to_numpy().ravel()

    logger.info("Train: %d  Test: %d", len(ids_train), len(ids_test))

    # ── Process all tables via TableTransformer (leak-free) ─────────────────
    included_tables = [t for t in ALL_TABLES if getattr(cfg.data, t).include]
    logger.info("Loading tables: %s", included_tables)

    raw_tables = {t: loader.load(t).collect() for t in included_tables}

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

    # Fit pipelines on train only; transform both train and test
    X_train_all, X_test_all, y_train, y_test, all_feature_names = table_transformer.fit_transform(
        tables=raw_tables,
        labels=labels_df,
        train_ids=set(ids_train),
        val_ids=set(ids_test),
    )
    logger.info(
        "Processed: %d total features  %d train  %d test",
        X_train_all.shape[1],
        X_train_all.shape[0],
        X_test_all.shape[0],
    )

    # ── Filter to selected features ──────────────────────────────────────────
    available = set(all_feature_names)
    feature_cols = [f for f in best_features if f in available]
    missing = [f for f in best_features if f not in available]
    if missing:
        logger.warning("%d selected features not found after processing: %s", len(missing), missing)
    logger.info("Using %d selected features", len(feature_cols))

    feat_idx = [all_feature_names.index(f) for f in feature_cols]
    X_train = X_train_all[:, feat_idx]
    X_test = X_test_all[:, feat_idx]

    # ── Find optimal threshold via CV on train ────────────────────────────────
    threshold_selector = SimpleThresholdSelector(metric="f1", direction="maximize")

    # Quick CV to get pooled proba for threshold selection
    from credit_risk.models.model_factory import get_factory as _get_factory

    model_factory = _get_factory(best_model_name, cfg.model.x_transform, cfg.model.nan_fill)

    # Collect OOF predictions for threshold selection
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg.splitter.n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = model_factory(**model_params).build_model_pipeline()
        m.fit(X_train[train_idx], y_train[train_idx])
        oof_proba[val_idx] = m.predict_proba(X_train[val_idx])

    optimal_threshold = threshold_selector.select(y_train, oof_proba)
    logger.info("Optimal threshold (CV F1): %.3f", optimal_threshold)

    # ── Train final model on full train set ──────────────────────────────────
    trainer = FinalModelTrainer(model_factory)
    result = trainer.train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        model_params,
        optimal_threshold=optimal_threshold,
    )

    logger.info("=" * 60)
    logger.info("Test ROC AUC:   %.4f", result.test_roc_auc)
    logger.info("Test F1:        %.4f", result.test_f1)
    logger.info("Test Recall:    %.4f", result.test_recall)
    logger.info("Test Precision: %.4f", result.test_precision)
    logger.info("Threshold:      %.3f", result.optimal_threshold)
    logger.info("=" * 60)

    # ── Get fitted model for plots and SHAP ──────────────────────────────────
    # FinalModelTrainer trains internally — re-fit on full train for artifacts
    final_model = model_factory(**model_params).build_model_pipeline()
    final_model.fit(X_train, y_train)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_path = cfg.output.models_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    plotter = ModelPlotter()
    y_pred_proba = final_model.predict_proba(X_test)
    plotter.plot_all(
        y_test, y_pred_proba, threshold=result.optimal_threshold, output_dir=plots_path
    )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    logger.info("Computing SHAP importance...")
    from credit_risk.models.importance.shap import SHAPImportance as _SI

    _imp = _SI(n_samples=300)
    X_sample = _imp._subsample(_imp._prepare(X_train))
    estimator = _imp._unwrap(final_model)
    if _imp._is_tree_model(estimator):
        import shap as _shap

        explainer = _shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        shap_values = None

    shap_plot_path = plots_path / "shap_summary.png"
    if shap_values is not None:
        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP plot saved: %s", shap_plot_path)

    # ── Save model + features ─────────────────────────────────────────────────
    model_dir = Path(cfg.output.models_path)
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"final_model_{run_mode}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("Model saved: %s", model_path)

    feature_path = model_dir / f"features_{run_mode}.json"
    feature_path.write_text(json.dumps(feature_cols, indent=2))
    logger.info("Features saved: %s", feature_path)

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with ml_logger.start_run(run_name=f"{run_mode}_final_model"):
        ml_logger.log_flat_config(cfg)
        ml_logger.log_params(
            {
                "model_type": best_model_name,
                "n_features": len(feature_cols),
                "optimal_threshold": optimal_threshold,
            }
        )
        ml_logger.log_params(model_params)
        ml_logger.log_metrics(
            {
                "test_roc_auc": result.test_roc_auc,
                "test_f1": result.test_f1,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "test_accuracy": result.test_accuracy,
                "optimal_threshold": result.optimal_threshold,
            }
        )
        ml_logger.log_model(final_model, "final_model")
        ml_logger.log_file_artifact(str(model_path))
        ml_logger.log_file_artifact(str(feature_path))
        if shap_values is not None:
            ml_logger.log_file_artifact(str(shap_plot_path))
        plotter.log_to_mlflow(True)

    logger.info("Final training complete")


if __name__ == "__main__":
    main()
