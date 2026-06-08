"""Model factory for creating classification pipelines with optional X transformers."""

from __future__ import annotations

import functools
from typing import Any, Protocol

from credit_risk_models import (
    EstimatorPipeline,
    EstimatorPipelineFactory,
    NanFillStrategy,
    NaNReplacer,
)
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
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
    "NaNReplacer",
    "NanFillStrategy",
]


class ModelFactory(Protocol):
    """Protocol for a callable factory that builds an EstimatorPipeline."""

    def __call__(self, **params: Any) -> EstimatorPipelineFactory:
        """Create a factory with given params."""

    def build_model_pipeline(self) -> EstimatorPipeline:
        """Build the model pipeline from a configured factory."""
        ...


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
    nan_fill: NanFillStrategy | None = None,
) -> ModelFactory:
    """Get a partial(EstimatorPipelineFactory, estimator_class=..., x_transformer=...).

    Args:
        model_type: One of "lgbm", "xgboost", "catboost", "hist_gbm",
            "random_forest", "extra_trees", "gradient_boosting", "lr",
            "ridge", "svm".
        x_transform: X transformation - "none", "standard", "min_max",
            "robust", "power", "quantile", "normalize".
        nan_fill: NaN replacement strategy. ``float`` replaces all NaN/inf
            with that constant. ``"median_mode"`` learns per-column medians
            at fit time and applies them at transform time. ``None`` skips
            NaN replacement entirely.

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
        nan_fill=nan_fill,
    )
