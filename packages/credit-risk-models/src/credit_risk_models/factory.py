"""EstimatorPipelineFactory — builds model pipelines with optional X transform and NaN handling."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from credit_risk_models.estimator_pipeline import ESTIMATOR_STEP, EstimatorPipeline
from credit_risk_models.nan_replacer import NanFillStrategy, NaNReplacer

X_STEP = "x_transform"
NAN_STEP = "nan_replace"


class EstimatorPipelineFactory:
    """Single generic factory for all classifier types.

    Supports X transformation via class methods or direct instantiation.

    Usage:
        factory = EstimatorPipelineFactory(estimator_class=LGBMClassifier, x_transformer=None)
        model = factory.build_model_pipeline()

        factory = EstimatorPipelineFactory(
            estimator_class=LogisticRegression, x_transformer=StandardScaler()
        )
        model = factory.build_model_pipeline()
    """

    def __init__(
        self,
        estimator_class: type[BaseEstimator],
        x_transformer: Any | None = None,
        nan_fill: NanFillStrategy | None = None,
        **params: Any,
    ):
        """Initialize the factory.

        Args:
            estimator_class: sklearn estimator class (e.g., LGBMClassifier).
            x_transformer: Optional sklearn transformer for X preprocessing.
                           None means no transform step (raw features).
            nan_fill: NaN replacement strategy. ``float`` replaces all NaN/inf
                      with that constant. ``"median_mode"`` learns per-column
                      medians at fit time and applies them at transform time.
                      ``None`` skips the step entirely.
            **params: Fixed parameters for the estimator (e.g. n_jobs=-1).
        """
        self.estimator_class = estimator_class
        self.x_transformer = x_transformer
        self.nan_fill = nan_fill
        self._params = params

    def build_model_pipeline(self) -> EstimatorPipeline:
        """Build the complete model pipeline.

        Returns:
            EstimatorPipeline with optional nan_replace, x_transform, and
            estimator steps, in that order.
        """
        params = self._params.copy()
        self._enforce_lgbm_constraints(params)
        estimator = self.estimator_class(**params)

        steps: list[tuple[str, Any]] = []
        if self.nan_fill is not None:
            steps.append((NAN_STEP, NaNReplacer(fill_value=self.nan_fill)))
        if self.x_transformer is not None:
            steps.append((X_STEP, self.x_transformer))
        steps.append((ESTIMATOR_STEP, estimator))

        return EstimatorPipeline(steps)

    @staticmethod
    def _enforce_lgbm_constraints(params: dict[str, Any]) -> None:
        if "num_leaves" in params and "max_depth" in params:
            max_allowed = 2 ** params["max_depth"] - 1
            if params["num_leaves"] > max_allowed:
                params["num_leaves"] = max_allowed

    @classmethod
    def raw(cls, estimator_class: type[BaseEstimator], **params: Any) -> EstimatorPipelineFactory:
        """No X transformation (raw features)."""
        return cls(estimator_class=estimator_class, x_transformer=None, **params)

    @classmethod
    def scaled(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> EstimatorPipelineFactory:
        """Standard scaling on X."""
        return cls(estimator_class=estimator_class, x_transformer=StandardScaler(), **params)

    @classmethod
    def min_max(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> EstimatorPipelineFactory:
        """MinMax scaling on X."""
        return cls(estimator_class=estimator_class, x_transformer=MinMaxScaler(), **params)

    @classmethod
    def robust(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> EstimatorPipelineFactory:
        """Robust scaling on X using median and IQR."""
        return cls(estimator_class=estimator_class, x_transformer=RobustScaler(), **params)

    @classmethod
    def no_transform(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> EstimatorPipelineFactory:
        """Identity transform (pass-through)."""
        return cls(
            estimator_class=estimator_class,
            x_transformer=FunctionTransformer(validate=False),
            **params,
        )
