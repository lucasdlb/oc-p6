"""Optuna hyperparameter tuner using CrossValidator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from credit_risk.config import TuningConfig
from credit_risk.models.cross_validator import CrossValidator, CVMetrics
from credit_risk.models.metrics import ClassificationMetrics, ClassificationRankingMetrics
from credit_risk.models.model_factory import get_factory
from credit_risk.models.param_spaces import _SUGGEST_FN, suggest_params
from credit_risk.models.splitter import Splitter

if TYPE_CHECKING:
    from credit_risk.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class ManyModelOptunaTuner:
    """Sequential Optuna tuner - optimizes one model at a time.

    Runs all trials for a single model type before moving to the next,
    making it easier to compare and debug than mixed model optimization.

    Usage:
        tuner = ManyModelOptunaTuner(
            splitter=splitter,
            metrics=metrics,
            tuning_config=cfg.tuning,
        )

        study = tuner.optimize_sequential(X, y, ["lgbm", "random_forest", "lr"])
    """

    def __init__(
        self,
        splitter: Splitter,
        metrics: ClassificationRankingMetrics | ClassificationMetrics,
        tuning_config: TuningConfig | None = None,
        mlflow_logging: bool = False,
    ):
        """Initialize ManyModelOptunaTuner."""
        self._splitter = splitter
        self._metrics = metrics
        self._config = tuning_config or TuningConfig()
        self._mlflow_logging = mlflow_logging
        self._results: dict[str, dict[str, Any]] = {}

    def optimize_sequential(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Run sequential Optuna optimization - one model at a time.

        Args:
            X: Feature matrix
            y: Target vector
            model_names: List of model names to tune sequentially

        Returns:
            Dict mapping model_name -> {best_value, best_params, study}
        """
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

            study = self._run_study(X, y, model_name, factory)

            self._results[model_name] = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study": study,
            }

            logger.info(f"{model_name} best ROC AUC: {study.best_value:.4f}")
            logger.info(f"{model_name} best params: {study.best_params}")

        return self._results

    def _run_study(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        factory: ModelFactory,
    ) -> optuna.Study:
        """Run Optuna study for a single model."""
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

            validator = CrossValidator(
                splitter=self._splitter,
                model_factory=factory,
                importance_strategy=None,
                verbose=False,
            )

            result = validator.validate(X, y, model_params=params)
            cv_scores = CVMetrics.compute(result, self._metrics)
            score = cv_scores.mean_scores.get("roc_auc", 0.0)

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
