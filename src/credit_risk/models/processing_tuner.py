"""Zero-leakage Optuna hyperparameter tuner using ProcessingCV.

Each Optuna trial runs a full ProcessingCV.validate() call, which performs
per-fold fit/transform of all tables via TableTransformer — no data leaks
between train and validation in any fold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import optuna

from credit_risk.config import TuningConfig
from credit_risk.models.cross_validator import CVMetrics
from credit_risk.models.metrics import ClassificationMetrics, ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.param_spaces import _SUGGEST_FN, suggest_params
from credit_risk.models.splitter import Splitter
from credit_risk.pipeline.cv_pipeline import ProcessingCV

if TYPE_CHECKING:
    import polars as pl

    from credit_risk.models.model_factory import ModelFactory
    from credit_risk.pipeline.table_transformer import TableTransformer

logger = logging.getLogger(__name__)


class ProcessingTuner:
    """Zero-leakage Optuna tuner using ProcessingCV.

    MLflow structure (all runs are nested under the caller's active run):

        {model_name}                        ← parent run per model type
        │  params: model, n_trials, x_transform, nan_fill, n_features
        │  metrics: best_roc_auc, best_pr_auc, best_gini, ...
        │  artifact: best_config.json
        │
        ├── trial_0                         ← nested run per trial
        │   params: all hyperparams
        │   metrics: mean_roc_auc, std_roc_auc, mean_pr_auc, ...
        ├── trial_1
        └── ...

    Usage:
        tuner = ProcessingTuner(
            table_transformer=table_transformer,
            splitter=splitter,
            tuning_config=cfg.tuning,
            metrics=ClassificationRankingMetrics(),
        )
        results = tuner.optimize(raw_tables, labels_df, feature_mask=features)
    """

    def __init__(
        self,
        table_transformer: "TableTransformer",
        splitter: Splitter,
        tuning_config: TuningConfig,
        metrics: ClassificationRankingMetrics | ClassificationMetrics,
    ):
        """Initialize ProcessingTuner.

        Args:
            table_transformer: TableTransformer for per-fold table processing.
            splitter: Train/test splitter (e.g. TrainTestCVSplitter).
            tuning_config: TuningConfig with n_trials, models, x_transform, etc.
            metrics: Metrics to evaluate per fold.
        """
        self._table_transformer = table_transformer
        self._splitter = splitter
        self._config = tuning_config
        self._metrics = metrics
        self._results: dict[str, dict[str, Any]] = {}

    def optimize(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_names: list[str] | None = None,
        feature_mask: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run sequential Optuna optimization across model types.

        Each model type gets a nested MLflow parent run.  Each trial within
        that model type gets a further nested run.

        Args:
            tables: Mapping of table name -> raw Polars DataFrame.
            labels: Polars DataFrame with id_column and target_column.
            model_names: Model types to tune. Defaults to cfg.tuning.models.
            feature_mask: Feature names to restrict CV to (from rfe_cv output).
                None means all features.

        Returns:
            Dict mapping model_name -> {best_value, best_params, study,
            best_scores}
        """
        model_names = model_names or self._config.models
        self._results = {}

        for model_name in model_names:
            logger.info("=" * 60)
            logger.info(f"Optimizing model: {model_name}")
            logger.info("=" * 60)

            if model_name not in _SUGGEST_FN:
                logger.warning(f"No param space for {model_name}, skipping")
                continue

            factory = get_factory(
                model_name,
                x_transform=self._config.x_transform,
                nan_fill=self._config.nan_fill,
            )

            study, best_scores = self._run_study(tables, labels, model_name, factory, feature_mask)

            self._results[model_name] = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_scores": best_scores,
                "study": study,
            }

            logger.info(f"{model_name} best ROC AUC: {study.best_value:.4f}")

        return self._results

    def _run_study(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_name: str,
        factory: "ModelFactory",
        feature_mask: list[str] | None = None,
    ) -> tuple[optuna.Study, dict[str, float]]:
        """Run Optuna study for a single model type.

        Table processing happens once before Optuna starts via build_folds().
        Each trial only does model fit/predict on pre-built numpy arrays — no
        Polars, no pipelines, no redundant table reprocessing in workers.

        Returns (study, best_mean_scores).
        """
        import mlflow

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        cv = ProcessingCV(
            table_transformer=self._table_transformer,
            splitter=self._splitter,
            model_factory=factory,
            importance_strategy=None,
            verbose=True,
        )

        # ── Pre-build fold arrays once ────────────────────────────────────────
        logger.info("Pre-building fold arrays (table processing)...")
        folds = cv.build_folds(tables, labels, feature_mask=feature_mask)
        n_features = folds[0].X_train.shape[1] if folds else 0
        logger.info(
            "Fold arrays ready: %d folds, %d features, %d train samples",
            len(folds),
            n_features,
            folds[0].X_train.shape[0] if folds else 0,
        )

        # Swap to a non-verbose CV for the trials (no per-fold logging noise)
        cv_eval = ProcessingCV(
            table_transformer=self._table_transformer,
            splitter=self._splitter,
            model_factory=factory,
            importance_strategy=None,
            verbose=False,
        )

        study = optuna.create_study(
            study_name=f"{self._config.study_name}_{model_name}",
            direction=self._config.direction,
        )

        n_jobs = self._config.n_jobs

        # Determine Optuna trial parallelism based on model compute device.
        #
        # GPU VRAM footprint on RTX 2060 (6 GB) at 195 features:
        #   LightGBM (OpenCL)  ~140 MiB  → parallel trials safe
        #   XGBoost  (CUDA)    ~179 MiB  → parallel trials safe
        #   CatBoost (GPU)     ~5.4 GB   → only one trial at a time
        #
        # CPU models with n_jobs=-1 must be sequential to avoid oversubscription.
        try:
            sample = suggest_params(study.ask(), model_name)
            uses_lgbm_gpu = sample.get("device") == "gpu"  # OpenCL
            uses_xgb_gpu = sample.get("device") == "cuda"  # CUDA
            uses_catboost_gpu = sample.get("task_type") == "GPU"  # ~5.4 GB
            uses_all_cpu = sample.get("n_jobs", 1) == -1

            if uses_catboost_gpu:
                if n_jobs != 1:
                    logger.info(
                        "CatBoost task_type=GPU uses ~5.4 GB VRAM — forcing Optuna n_jobs=1."
                    )
                n_jobs = 1
            elif uses_lgbm_gpu or uses_xgb_gpu:
                if n_jobs != 1:
                    logger.info(
                        "Model GPU device — Optuna n_jobs=%d "
                        "(each trial runs in a separate process, fold arrays passed via closure).",
                        n_jobs,
                    )
            elif uses_all_cpu and n_jobs != 1:
                logger.info(
                    "Model n_jobs=-1 (CPU) — forcing Optuna n_jobs=1 to avoid CPU oversubscription."
                )
                n_jobs = 1
        except Exception:
            pass

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, model_name)

            # Use pre-built fold arrays — no table processing here
            result = cv_eval.validate_folds(folds, model_params=params)
            cv_scores = CVMetrics.compute(result, self._metrics)
            score = cv_scores.mean_scores.get("roc_auc", 0.0)

            # Store scores on trial attrs — safe across multiprocessing workers
            for k, v in cv_scores.mean_scores.items():
                trial.set_user_attr(f"mean_{k}", v)
            for k, v in cv_scores.std_scores.items():
                trial.set_user_attr(f"std_{k}", v)

            # One nested MLflow run per trial
            if mlflow.active_run():
                with mlflow.start_run(nested=True, run_name=f"{model_name}__trial_{trial.number}"):
                    mlflow.log_params({**params, "trial": trial.number})
                    flat_metrics = {f"mean_{k}": v for k, v in cv_scores.mean_scores.items()}
                    flat_metrics.update({f"std_{k}": v for k, v in cv_scores.std_scores.items()})
                    mlflow.log_metrics(flat_metrics)

            logger.info(
                "Trial %d: roc_auc=%.4f ± %.4f",
                trial.number,
                score,
                cv_scores.std_scores.get("roc_auc", 0.0),
            )

            return score

        # Model-type parent run
        with mlflow.start_run(nested=True, run_name=f"{model_name}"):
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "n_trials": self._config.n_trials,
                    "x_transform": self._config.x_transform,
                    "nan_fill": str(self._config.nan_fill),
                    "n_features": n_features,
                }
            )

            study.optimize(
                objective,
                n_trials=self._config.n_trials,
                timeout=self._config.timeout,
                n_jobs=n_jobs,
            )

            # Retrieve best trial scores from trial attrs (safe after optimize)
            best_trial = study.best_trial
            best_mean = {
                k.removeprefix("mean_"): v
                for k, v in best_trial.user_attrs.items()
                if k.startswith("mean_")
            }
            best_std = {
                k.removeprefix("std_"): v
                for k, v in best_trial.user_attrs.items()
                if k.startswith("std_")
            }

            # Log all best metrics on model parent run
            mlflow.log_metrics({f"best_{k}": v for k, v in best_mean.items()})
            mlflow.log_metrics({f"best_std_{k}": v for k, v in best_std.items()})

            best_config = {
                "model_type": model_name,
                "x_transform": self._config.x_transform,
                "nan_fill": self._config.nan_fill,
                "best_params": study.best_params,
                "best_metrics": {f"mean_{k}": v for k, v in best_mean.items()},
                "best_metrics_std": {f"std_{k}": v for k, v in best_std.items()},
                "n_features": n_features,
                "n_trials": self._config.n_trials,
            }
            mlflow.log_dict(best_config, "best_config.json")

        return study, best_mean
