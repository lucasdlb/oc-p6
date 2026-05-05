"""Zero-leakage Optuna hyperparameter tuner using ProcessingCV.

Each Optuna trial runs a full ProcessingCV.validate() call, which performs
per-fold fit/transform of all tables via TableTransformer — no data leaks
between train and validation in any fold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import optuna
import polars as pl

from credit_risk.config import TuningConfig
from credit_risk.models.cross_validator import CVMetrics
from credit_risk.models.metrics import ClassificationMetrics, ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.param_spaces import _SUGGEST_FN, suggest_params
from credit_risk.models.splitter import Splitter
from credit_risk.pipeline.cv_pipeline import ProcessingCV

if TYPE_CHECKING:
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

        Returns (study, best_mean_scores) where best_mean_scores contains all
        metrics from the best trial.
        """
        import mlflow

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        cv = ProcessingCV(
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

        # Most models use n_jobs=-1 (all cores). Running Optuna trials in
        # parallel on top of that would oversubscribe the CPU. Check whether
        # this model's fixed params include n_jobs=-1 and force Optuna to 1.
        try:
            sample = suggest_params(study.ask(), model_name)
            if sample.get("n_jobs", 1) == -1 and n_jobs != 1:
                logger.info(
                    "Model n_jobs=-1 — forcing Optuna n_jobs=1 to avoid CPU oversubscription."
                )
                n_jobs = 1
        except Exception:
            pass

        # Collect all trial scores keyed by trial number for post-hoc lookup
        trial_scores: dict[int, dict[str, float]] = {}
        trial_scores_std: dict[int, dict[str, float]] = {}

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, model_name)

            result = cv.validate(
                tables=tables,
                labels=labels,
                model_params=params,
                feature_mask=feature_mask,
            )
            cv_scores = CVMetrics.compute(result, self._metrics)
            score = cv_scores.mean_scores.get("roc_auc", 0.0)

            trial_scores[trial.number] = cv_scores.mean_scores
            trial_scores_std[trial.number] = cv_scores.std_scores

            # Store all metrics on the trial for Optuna's user_attrs
            for k, v in cv_scores.mean_scores.items():
                trial.set_user_attr(f"mean_{k}", v)
            for k, v in cv_scores.std_scores.items():
                trial.set_user_attr(f"std_{k}", v)

            # One nested MLflow run per trial
            if mlflow.active_run():
                with mlflow.start_run(nested=True, run_name=f"{model_name}__trial_{trial.number}"):
                    mlflow.log_params({**params, "trial": trial.number})
                    # Log mean and std for every metric
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
                    "n_features": len(feature_mask) if feature_mask else "all",
                }
            )

            study.optimize(
                objective,
                n_trials=self._config.n_trials,
                timeout=self._config.timeout,
                n_jobs=n_jobs,
            )

            # Best trial metrics
            best_trial_num = study.best_trial.number
            best_mean = trial_scores.get(best_trial_num, {})
            best_std = trial_scores_std.get(best_trial_num, {})

            # Log all best metrics on model parent run
            mlflow.log_metrics({f"best_{k}": v for k, v in best_mean.items()})
            mlflow.log_metrics({f"best_std_{k}": v for k, v in best_std.items()})

            # Save full best config as artifact
            best_config = {
                "model_type": model_name,
                "x_transform": self._config.x_transform,
                "nan_fill": self._config.nan_fill,
                "best_params": study.best_params,
                "best_metrics": {f"mean_{k}": v for k, v in best_mean.items()},
                "best_metrics_std": {f"std_{k}": v for k, v in best_std.items()},
                "n_features": len(feature_mask) if feature_mask else None,
                "n_trials": self._config.n_trials,
            }
            mlflow.log_dict(best_config, "best_config.json")

        return study, best_mean
