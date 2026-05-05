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

    Each Optuna trial calls ProcessingCV.validate(tables, labels, model_params),
    which performs fresh per-fold table processing via TableTransformer.
    No information from the validation set leaks into feature engineering.

    Usage:
        tuner = ProcessingTuner(
            table_transformer=table_transformer,
            splitter=splitter,
            tuning_config=cfg.tuning,
            metrics=ClassificationRankingMetrics(),
            mlflow_logging=True,
        )
        results = tuner.optimize(raw_tables, labels_df)
    """

    def __init__(
        self,
        table_transformer: TableTransformer,
        splitter: Splitter,
        tuning_config: TuningConfig,
        metrics: ClassificationRankingMetrics | ClassificationMetrics,
        mlflow_logging: bool = True,
    ):
        """Initialize ProcessingTuner.

        Args:
            table_transformer: TableTransformer for per-fold table processing.
            splitter: Train/test splitter (e.g. TrainTestCVSplitter).
            tuning_config: TuningConfig with n_trials, models, x_transform, etc.
            metrics: Metrics to evaluate per fold.
            mlflow_logging: Whether to log to MLflow.
        """
        self._table_transformer = table_transformer
        self._splitter = splitter
        self._config = tuning_config
        self._metrics = metrics
        self._mlflow_logging = mlflow_logging
        self._results: dict[str, dict[str, Any]] = {}

    def optimize(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run sequential Optuna optimization across model types.

        Args:
            tables: Mapping of table name -> raw Polars DataFrame.
            labels: Polars DataFrame with id_column and target_column.
            model_names: List of model names to tune. Defaults to cfg.tuning.models.

        Returns:
            Dict mapping model_name -> {best_value, best_params, study}
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

            study = self._run_study(tables, labels, model_name, factory)

            self._results[model_name] = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study": study,
            }

            logger.info(f"{model_name} best ROC AUC: {study.best_value:.4f}")

        return self._results

    def _run_study(
        self,
        tables: dict[str, pl.DataFrame],
        labels: pl.DataFrame,
        model_name: str,
        factory: ModelFactory,
    ) -> optuna.Study:
        """Run Optuna study for a single model type.

        A single ProcessingCV instance is created per model type and reused
        across all trials — the ProcessingCV itself is stateless (TableTransformer
        creates fresh pipelines per fold).
        """
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
        if self._mlflow_logging and n_jobs > 1:
            logger.warning("MLflow logging enabled, forcing n_jobs=1 to avoid race conditions")
            n_jobs = 1

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, model_name)

            result = cv.validate(tables=tables, labels=labels, model_params=params)
            cv_scores = CVMetrics.compute(result, self._metrics)
            score = cv_scores.mean_scores.get("roc_auc", 0.0)

            logger.info(f"CV score: {score:.4f}")

            trial.set_user_attr("score", score)
            for k, v in cv_scores.mean_scores.items():
                trial.set_user_attr(k, v)

            if self._mlflow_logging:
                import mlflow

                if mlflow.active_run():
                    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                        mlflow.log_params(params)
                        mlflow.log_metrics(cv_scores.mean_scores)

            return score

        if self._mlflow_logging:
            import mlflow

            with mlflow.start_run(nested=True, run_name=model_name):
                mlflow.log_params({"model": model_name, "n_trials": self._config.n_trials})
                study.optimize(
                    objective,
                    n_trials=self._config.n_trials,
                    timeout=self._config.timeout,
                    n_jobs=n_jobs,
                )
                mlflow.log_metric("best_roc_auc", study.best_value)
        else:
            study.optimize(
                objective,
                n_trials=self._config.n_trials,
                timeout=self._config.timeout,
                n_jobs=n_jobs,
            )

        return study
