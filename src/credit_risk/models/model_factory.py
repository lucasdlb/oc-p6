"""Model factory for creating classification pipelines with optional X transformers."""

from __future__ import annotations

import functools
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "EstimatorPipeline",
    "EstimatorPipelineFactory",
    "get_factory",
    "MODEL_REGISTRY",
    "ModelFactory",
]

X_STEP = "x_transform"
ESTIMATOR_STEP = "estimator"
NAN_STEP = "nan_replace"


NanFillStrategy = float | Literal["median_mode"]


class NaNReplacer(TransformerMixin, BaseEstimator):
    """Replace NaN and inf values using a per-column or constant strategy.

    Two strategies are supported, controlled by ``fill_value``:

    - **Constant** (``fill_value`` is a ``float``): replaces every NaN/inf
      with the given scalar.  ``fit`` is a no-op; the replacement is
      stateless.
    - **Median/mode** (``fill_value="median_mode"``): stateful imputer.
      ``fit`` learns, for each column, the median (numeric) or the most
      frequent value (non-numeric / object).  ``transform`` applies those
      per-column fill values and also replaces ±inf with the same value.

    In both strategies the fitted fill values are stored in
    ``fill_values_`` (a 1-D numpy array of ``float64``, length = number of
    columns seen at fit time).

    Args:
        fill_value: Scalar constant or ``"median_mode"`` strategy selector.
    """

    def __init__(self, fill_value: NanFillStrategy = 0.0):
        self.fill_value = fill_value

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NaNReplacer:
        """Learn per-column fill values from training data.

        Args:
            X: 2-D numeric array of shape (n_samples, n_features).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        if self.fill_value == "median_mode":
            fills = np.empty(n_features, dtype=float)
            for j in range(n_features):
                col = X[:, j]
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    fills[j] = 0.0
                else:
                    fills[j] = float(np.median(finite))
        else:
            fills = np.full(n_features, float(self.fill_value))

        self.fill_values_: np.ndarray = fills
        self.n_features_in_: int = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN and ±inf values using fitted fill values.

        Args:
            X: 2-D numeric array of shape (n_samples, n_features).

        Returns:
            Array with the same shape as ``X`` with NaN/inf replaced.
        """
        check_is_fitted(self, "fill_values_")
        X = np.array(X, dtype=float)

        for j, fill in enumerate(self.fill_values_):
            col = X[:, j]
            mask = ~np.isfinite(col)
            if mask.any():
                col[mask] = fill

        return X


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

        Falls back to decision_function for estimators without predict_proba
        (e.g. RidgeClassifier, SVC with probability=False).
        """
        try:
            result = super().predict_proba(X, **params)
            if result.max() > 1.0 or result.min() < 0.0:
                from scipy.special import expit

                result = expit(result)
            return result[:, 1]
        except AttributeError:
            from scipy.special import expit

            scores = super().decision_function(X, **params)
            return expit(scores)

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
                      medians at fit time. ``None`` skips the step entirely.
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
