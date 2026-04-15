#!/usr/bin/env python
"""Final model training using selected features and tuned hyperparameters.

Loads selected features from FeatureStore, loads best params from MLflow run,
then trains and saves final model using FinalModelTrainer.

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

from credit_risk.config import cfg
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoding import CategoricalEncoder
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.features.aggregator import DataAggregator
from credit_risk.features.store import FeatureStore
from credit_risk.features.transformer import DataTransformer
from credit_risk.interpret.shap_explainer import ShapExplainer
from credit_risk.models.cross_validator import LGBMFactory
from credit_risk.models.final_model import FinalModelTrainer
from credit_risk.models.plotter import ModelPlotter
from credit_risk.models.resampler import create_resampler
from credit_risk.models.splitter import TrainTestCVSplitter
from credit_risk.models.threshold_selector import CVThresholdSelector

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
    "credit_card_balance",
    "installments_payments",
]

run_mode = cfg.run.mode
logger.info(f"Running final training in mode: {run_mode}")

mlflow.set_tracking_uri(f"sqlite:///{cfg.output.mlflow_db_path}")
mlflow.set_experiment(f"final_train_{cfg.run.mode}-weighted_threshold-fix")

sample_frac = cfg.run.sample_fraction
if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of data for {run_mode} mode")

logger.info("Loading data...")
loader = PLLazyDataLoader()
labels = loader.load_labels()

cleaner = DataCleaner()
imputer = DataImputer()
aggregator = DataAggregator()
transformer = DataTransformer()

store = FeatureStore(root=cfg.output.features_path)

all_selected = []
saved_records = {}
for table in TABLES:
    # feature_name = f"{table}_{run_mode}"
    feature_name = f"{table}_prod"
    try:
        features = store.load(feature_name)
        record = store.load_record(feature_name)
        all_selected.extend(features)
        saved_records[table] = record
        logger.info(f"Loaded {len(features)} features from {feature_name}")
    except KeyError:
        logger.warning(f"No saved features for {feature_name}")

all_selected = list(set(all_selected))
logger.info(f"Total selected features: {len(all_selected)}")

features_list = []
for table in TABLES:
    logger.info(f"Processing table: {table}")

    df = loader.load(table)
    df = df.collect() if hasattr(df, "collect") else df

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac)

    logger.info(f"  Loaded: {df.height} rows, {df.width} cols")

    df = cleaner.clean(df, table, method=cfg.processing.cleaning)
    df = imputer.impute(df, table, method=cfg.processing.imputation)
    df = aggregator.aggregate(df.lazy(), table, method=cfg.processing.aggregation).collect()
    df = transformer.transform(df, table=table, method=cfg.processing.encoding)

    id_cols = [c for c in df.columns if c.startswith("SK_ID")]
    feature_cols = [c for c in df.columns if c not in id_cols + ["TARGET"]]
    df_select = df.select(["SK_ID_CURR"] + feature_cols)
    if df_select.schema["SK_ID_CURR"] != pl.Int64:
        df_select = df_select.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    features_list.append(df_select)

logger.info(f"Joining {len(features_list)} tables with labels...")

combined = labels.lazy()
if combined.schema["SK_ID_CURR"] != pl.Int64:
    combined = combined.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
for i, df in enumerate(features_list):
    combined = combined.join(df.lazy(), on="SK_ID_CURR", how="left", suffix=f"_{i}")

combined = combined.collect()

logger.info(f"Combined: {combined.height} rows, {combined.width} cols")

if sample_frac < 1.0:
    logger.info(f"Sampling {sample_frac * 100}% of combined data")
    combined = combined.sample(fraction=sample_frac, seed=cfg.run.random_state)
    logger.info(f"Sampled: {combined.height} rows")

target_col = cfg.data.target.column
id_col = cfg.data.target.id_column

non_numeric = [c for c in combined.columns if combined.schema[c] == pl.String]
if non_numeric and cfg.processing.encoding != "none":
    logger.info(f"Encoding {len(non_numeric)} categorical columns")
    encoder = CategoricalEncoder()
    combined = encoder.fit_transform(combined)
    logger.info(f"Encoded: {combined.height} rows, {combined.width} cols")

available_cols = set(combined.columns) - {target_col, id_col}
feature_cols = [c for c in all_selected if c in available_cols]
logger.info(f"Using {len(feature_cols)} selected features")

X = combined.select(feature_cols).to_pandas()
y = combined.select(target_col).to_numpy().ravel()

# X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
X = X.to_numpy(dtype=np.float64)

logger.info(f"Data shape: {X.shape}, Features: {len(feature_cols)}")

resampler = None
if cfg.resampling.enabled:
    method = cfg.resampling.method
    kwargs = {
        "sampling_strategy": cfg.resampling.sampling_strategy,
        "random_state": cfg.resampling.random_state,
    }
    if method == "smote":
        kwargs["k_neighbors"] = cfg.resampling.k_neighbors
    resampler = create_resampler(method, **kwargs)
    logger.info(
        f"Using {cfg.resampling.method} resampling: "
        f"sampling_strategy={cfg.resampling.sampling_strategy}"
    )

splitter = TrainTestCVSplitter(
    test_size=cfg.splitter.test_size,
    n_splits=cfg.splitter.n_splits,
    random_state=cfg.splitter.random_state,
    stratify=True,
)

if resampler is not None:
    splitter.set_resampler(resampler)

X_train, X_test, y_train, y_test = splitter.split_train_test(X, y)
logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

best_params = None
best_model_name = None

mlflow_active = False
try:
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp:
        runs = mlflow.search_runs(
            [exp.experiment_id],
            "metrics.best_roc_auc > 0",
            max_results=20,
            order_by=["metrics.best_roc_auc DESC"],
        )

        # Find run with actual tuned params (not just model name)
        for _, best_run in runs.iterrows():
            run_id = best_run.run_id
            client = mlflow.MlflowClient()
            run = client.get_run(run_id)
            params = dict(run.data.params)

            # Check if this run has actual tuning params (e.g., n_estimators, learning_rate)
            if "n_estimators" in params or "learning_rate" in params:
                best_roc_auc = best_run.get("metrics.best_roc_auc", 0)
                best_params = params
                best_model_name = params.get("best_model", "lgbm")
                mlflow_active = True
                logger.info(f"Found best run: {run_id}, ROC AUC: {best_roc_auc:.4f}")
                logger.info(f"Best model: {best_model_name}")
                logger.info(f"Best params: {best_params}")
                break
            else:
                logger.debug(f"Skipping run {run_id[:8]} - no tuning params")
except Exception as e:
    logger.warning(f"Could not load from MLflow: {e}")

if best_params is None:
    app_record = saved_records.get("application", {})
    meta = app_record.get("meta", {})
    best_params = meta.get("model_params", {})
    best_model_name = "lgbm"
    logger.info(f"Using params from feature store: {best_params}")

model_params = {
    k: v for k, v in best_params.items() if k not in ("verbose", "run_mode", "models", "best_model")
}

# Convert string params to proper types
for k, v in model_params.items():
    if isinstance(v, str):
        try:
            if "." in v:
                model_params[k] = float(v)
            else:
                model_params[k] = int(v)
        except ValueError:
            pass  # Keep as string

model_params["verbose"] = -1
model_params["is_unbalance"] = True

if "class_weight" in best_params:
    model_params["class_weight"] = best_params["class_weight"]

logger.info(f"Training final model with params: {model_params}")

model_factory = LGBMFactory()
threshold_selector = CVThresholdSelector(
    splitter,
    model_factory,
    direction="maximize",
    metric="f1",
    # custom_func=create_cost_sensitive_score(fn_weight=2),
)

trainer = FinalModelTrainer(model_factory, threshold_selector=threshold_selector)
result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols, model_params)

logger.info(f"Test ROC AUC: {result.test_roc_auc:.4f}")
logger.info(f"Test F1: {result.test_f1:.4f}")
logger.info(f"Test Precision: {result.test_precision:.4f}")
logger.info(f"Test Recall: {result.test_recall:.4f}")
logger.info(f"Optimal threshold: {result.optimal_threshold:.2f}")

y_pred_proba = (
    model_factory.create(**model_params).fit(X_train, y_train).predict_proba(X_test)[:, 1]
)

plotter = ModelPlotter()
plots_path = cfg.output.models_path / "plots"
plotter.plot_all(y_test, y_pred_proba, threshold=result.optimal_threshold, output_dir=plots_path)

model_dir = Path(cfg.output.models_path)
model_dir.mkdir(exist_ok=True)
model_path = model_dir / f"final_model_{run_mode}.pkl"
final_model = model_factory.create(**model_params).fit(X_train, y_train)
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
logger.info(f"Model saved to {model_path}")

feature_path = model_dir / f"features_{run_mode}.json"
with open(feature_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
logger.info(f"Features saved to {feature_path}")

logger.info("Computing SHAP feature importance...")
numpy_X = X_train.numpy() if hasattr(X_train, "numpy") else X_train

shap_exp = ShapExplainer()
shap_exp.fit(final_model, numpy_X)
shap_vals, mean_abs = shap_exp.global_importance(numpy_X, n_samples=300)

shap_plot_path = plots_path / "shap_summary.png"
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, numpy_X[:300], feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"SHAP plot saved to {shap_plot_path}")

with mlflow.start_run(run_name=f"{run_mode}_final_model"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_roc_auc", result.test_roc_auc)
    mlflow.log_metric("test_f1", result.test_f1)
    mlflow.log_metric("test_precision", result.test_precision)
    mlflow.log_metric("test_recall", result.test_recall)
    mlflow.log_metric("optimal_threshold", result.optimal_threshold)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.sklearn.log_model(final_model, "final_model")
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(feature_path)
    mlflow.log_artifact(shap_plot_path)
    plotter.log_to_mlflow(True)

logger.info("Final training complete")
