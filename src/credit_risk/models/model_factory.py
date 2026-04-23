"""Model factory for creating classification pipelines with optional X transformers."""

from __future__ import annotations

import functools
from typing import Any, Protocol

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "EstimatorPipeline",
    "EstimatorPipelineFactory",
    "get_factory",
    "MODEL_REGISTRY",
    "ModelFactory",
]

X_STEP = "x_transform"
ESTIMATOR_STEP = "estimator"


class ModelFactory(Protocol):
    """Protocol for a callable factory that builds an EstimatorPipeline."""

    def __call__(self, **params: Any) -> EstimatorPipelineFactory:
        """Create a factory with given params."""

    def build_model_pipeline(self) -> EstimatorPipeline:
        """Build the model pipeline from a configured factory."""
        ...


class EstimatorPipeline(Pipeline):
    """Pipeline with x_transform and estimator steps.

    Provides utility methods for accessing the underlying estimator,
    prediction methods, and feature importances.
    """

    def get_final_estimator(self) -> BaseEstimator:
        """Return the actual estimator (unwrapped from any wrapper)."""
        return self.named_steps[ESTIMATOR_STEP]

    def predict_proba(self, X, **params):
        """Predict class probabilities, returning positive-class scores (1D).

        Handles LightGBM's cross_entropy_lambda objective which returns raw
        logit scores instead of probabilities — applies sigmoid when output
        exceeds the [0, 1] range.
        """
        result = super().predict_proba(X, **params)
        if result.max() > 1.0 or result.min() < 0.0:
            from scipy.special import expit
            result = expit(result)
        return result[:, 1]

    def get_feature_importances(self):
        """Get feature importances from the estimator.

        Handles both tree-based models (feature_importances_) and
        linear models (abs(coef_) averaged across classes).

        Returns:
            Feature importances as numpy array, or None if not available.
        """
        estimator = self.get_final_estimator()
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef)
        return None


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
        **params: Any,
    ):
        """Initialize the factory.

        Args:
            estimator_class: sklearn estimator class (e.g., LGBMClassifier).
            x_transformer: Optional sklearn transformer for X preprocessing.
                           None means no transform step (raw features).
            **params: Fixed parameters for the estimator (e.g. n_jobs=-1).
        """
        self.estimator_class = estimator_class
        self.x_transformer = x_transformer
        self._params = params

    def build_model_pipeline(self) -> EstimatorPipeline:
        """Build the complete model pipeline.

        Returns:
            EstimatorPipeline with x_transform (if x_transformer set) and estimator steps.
        """
        params = self._params.copy()
        self._enforce_lgbm_constraints(params)
        estimator = self.estimator_class(**params)

        if self.x_transformer is None:
            return EstimatorPipeline([(ESTIMATOR_STEP, estimator)])

        return EstimatorPipeline(
            [
                (X_STEP, self.x_transformer),
                (ESTIMATOR_STEP, estimator),
            ]
        )

    @staticmethod
    def _enforce_lgbm_constraints(params: dict[str, Any]) -> None:
        if "num_leaves" in params and "max_depth" in params:
            max_allowed = 2 ** params["max_depth"] - 1
            if params["num_leaves"] > max_allowed:
                params["num_leaves"] = max_allowed

    # -------------------------------------------------------------------------
    # Class methods: named constructors for common X transform configurations
    # -------------------------------------------------------------------------

    @classmethod
    def raw(cls, estimator_class: type[BaseEstimator], **params: Any) -> "EstimatorPipelineFactory":
        """No X transformation (raw features)."""
        return cls(estimator_class=estimator_class, x_transformer=None, **params)

    @classmethod
    def scaled(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> "EstimatorPipelineFactory":
        """Standard scaling on X."""
        return cls(estimator_class=estimator_class, x_transformer=StandardScaler(), **params)

    @classmethod
    def min_max(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> "EstimatorPipelineFactory":
        """MinMax scaling on X."""
        return cls(estimator_class=estimator_class, x_transformer=MinMaxScaler(), **params)

    @classmethod
    def robust(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> "EstimatorPipelineFactory":
        """Robust scaling on X using median and IQR."""
        return cls(estimator_class=estimator_class, x_transformer=RobustScaler(), **params)

    @classmethod
    def no_transform(
        cls, estimator_class: type[BaseEstimator], **params: Any
    ) -> "EstimatorPipelineFactory":
        """Identity transform (pass-through)."""
        return cls(
            estimator_class=estimator_class,
            x_transformer=FunctionTransformer(validate=False),
            **params,
        )


MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {}

_XFORMER_REGISTRY: dict[str, Any | None] = {}


def _register_models() -> None:
    """Register all model and transformer factories (called on first get_factory call)."""
    global MODEL_REGISTRY, _XFORMER_REGISTRY
    if MODEL_REGISTRY:
        return

    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    MODEL_REGISTRY = {
        "lgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
        "catboost": CatBoostClassifier,
        "hist_gbm": HistGradientBoostingClassifier,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "lr": LogisticRegression,
        "ridge": RidgeClassifier,
        "svm": SVC,
    }

    _XFORMER_REGISTRY = {
        "none": None,
        "standard": StandardScaler(),
        "min_max": MinMaxScaler(),
        "robust": RobustScaler(),
        "power": PowerTransformer(method="yeo-johnson"),
        "quantile": QuantileTransformer(output_distribution="normal", random_state=42),
        "normalize": Normalizer(),
    }


def get_factory(
    model_type: str,
    x_transform: str = "none",
) -> ModelFactory:
    """Get a partial(EstimatorPipelineFactory, estimator_class=..., x_transformer=...).

    Args:
        model_type: One of "lgbm", "xgboost", "catboost", "hist_gbm",
            "random_forest", "extra_trees", "gradient_boosting", "lr",
            "ridge", "svm".
        x_transform: X transformation - "none", "standard", "min_max",
            "robust", "power", "quantile", "normalize".

    Returns:
        functools.partial ready to call with **model_params.

    Usage:
        factory = get_factory("lgbm", "none")
        model = factory(n_estimators=500, max_depth=5).build_model_pipeline()
    """
    _register_models()

    estimator_class = MODEL_REGISTRY.get(model_type)
    if estimator_class is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    xformer = _XFORMER_REGISTRY.get(x_transform, Ellipsis)
    if xformer is Ellipsis:
        available = ", ".join(_XFORMER_REGISTRY.keys())
        raise ValueError(f"Unknown x_transform '{x_transform}'. Available: {available}")

    return functools.partial(
        EstimatorPipelineFactory,
        estimator_class=estimator_class,
        x_transformer=xformer,
    )
