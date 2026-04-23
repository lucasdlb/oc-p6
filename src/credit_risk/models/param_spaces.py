"""Hyperparameter search spaces for Optuna — binary classification on Home Credit.

Design rules
------------
- Seeds fixed to 42, never tuned.
- Operational params (n_jobs, verbose) excluded.
- All conditionals encoded directly in per-model suggest functions — no
  flat dict that pretends params are independent when they aren't.
- class imbalance: never "balanced" or None — always explicit pos weight
  or explicit class weight dict bracketing the true ~8% positive rate.

Public API
----------
suggest_params(trial, model_name) -> dict
    Single entry point used by the Optuna objective. Returns a fully
    valid param dict for the given model with all conditionals resolved.

FIXED
    Constants injected at model instantiation time (not searched).
"""

from __future__ import annotations

from typing import Any

import optuna

# ── Constants ─────────────────────────────────────────────────────────────────

FIXED: dict[str, Any] = {
    "random_state": 42,
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Per-model suggest functions ───────────────────────────────────────────────


def _suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Core
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 512),
        # num_leaves constrained to <= 2^max_depth — enforce at fit time in factory
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.0, 10.0),
        # Sampling
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # Regularisation
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 500),
        "min_data_in_bin": trial.suggest_int("min_data_in_bin", 1, 500),
        "max_bin": trial.suggest_int("max_bin", 50, 1000),
        # Imbalance — explicit pos weight, never is_unbalance (they conflict)
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
        # Objective / metric
        "objective": trial.suggest_categorical(
            "objective", ["binary", "cross_entropy", "cross_entropy_lambda"]
        ),
        "metric": trial.suggest_categorical("metric", ["auc", "binary_logloss"]),
    }

    # ── Boosting type with its conditionals ───────────────────────────────────
    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "rf"])
    params["boosting_type"] = boosting

    if boosting == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.0, 0.5)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 1.0)
        params["uniform_drop"] = trial.suggest_categorical("uniform_drop", [True, False])

    # bagging: required for rf, optional for gbdt/dart
    if boosting == "rf":
        # rf requires bagging — force freq >= 1 and fraction < 1
        params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 0.99)
    else:
        bagging_freq = trial.suggest_int("bagging_freq", 0, 10)
        params["bagging_freq"] = bagging_freq
        if bagging_freq > 0:
            params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)

    return params


def _suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Core
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        # Sampling
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        # Regularisation
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 64, 1024),
        # Tree method
        "tree_method": trial.suggest_categorical("tree_method", ["hist", "approx"]),
        # Imbalance
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
    }

    # max_leaves only meaningful with lossguide
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params["grow_policy"] = grow_policy
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 15, 511)

    return params


def _suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Core
        "iterations": trial.suggest_int("iterations", 500, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 20.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "border_count": trial.suggest_int("border_count", 32, 255),
        # Regularisation
        "random_strength": trial.suggest_float("random_strength", 0.01, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        # Leaf estimation
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 20),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient", "Exact"]
        ),
        # Missing values — Forbidden excluded (Home Credit has significant missingness)
        "nan_mode": trial.suggest_categorical("nan_mode", ["Min", "Max"]),
        # Imbalance
        "class_weights": {0: 1, 1: trial.suggest_float("pos_weight_cat", 5.0, 15.0)},
    }

    # grow_policy must be suggested BEFORE boosting_type since Ordered boosting
    # only supports SymmetricTree grow policy
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params["grow_policy"] = grow_policy

    # boosting_type depends on grow_policy
    if grow_policy == "SymmetricTree":
        boosting_type = trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])
    else:
        boosting_type = trial.suggest_categorical("boosting_type", ["Plain"])
    params["boosting_type"] = boosting_type

    # bootstrap_type drives bagging_temperature vs subsample
    bootstrap = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "No"])
    params["bootstrap_type"] = bootstrap
    if bootstrap == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
    elif bootstrap in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)

    if grow_policy == "Lossguide":
        params["score_function"] = trial.suggest_categorical("score_function", ["Cosine", "L2"])
    else:
        params["score_function"] = trial.suggest_categorical("score_function", ["Cosine", "L2"])

    return params


def _suggest_hist_gbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 511),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 1000),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.001, 20.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.05, 1.0),
        "max_bins": trial.suggest_int("max_bins", 32, 255),
        "validation_fraction": trial.suggest_float("validation_fraction", 0.05, 0.3),
        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 3, 50),
        "scoring": trial.suggest_categorical("scoring", ["loss", "roc_auc"]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }


def _suggest_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 511),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1.0),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1.0, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }

    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    params["bootstrap"] = bootstrap
    if bootstrap:
        params["oob_score"] = trial.suggest_categorical("oob_score", [True, False])

    return params


def _suggest_extra_trees(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 511),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1.0, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }
    return params


def _suggest_gradient_boosting(trial: optuna.Trial) -> dict[str, Any]:
    # GradientBoostingClassifier does not support class_weight —
    # use sample_weight at fit() time instead.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 0.9]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 511),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1.0),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1.0, log=True),
        "validation_fraction": trial.suggest_float("validation_fraction", 0.05, 0.3),
        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 3, 50),
        "tol": trial.suggest_float("tol", 1e-8, 1e-2, log=True),
        "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"]),
    }


def _suggest_lr(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 5000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }

    # solver / penalty are tightly coupled
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    params["solver"] = solver

    if solver == "saga":
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        params["penalty"] = penalty
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif solver == "liblinear":
        params["penalty"] = trial.suggest_categorical("penalty", ["l2", "l1"])
    else:  # lbfgs
        params["penalty"] = "l2"

    return params


def _suggest_ridge(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
        ),
        "max_iter": trial.suggest_int("max_iter", 200, 5000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }


def _suggest_svm(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "shrinking": trial.suggest_categorical("shrinking", [True, False]),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "cache_size": trial.suggest_int("cache_size", 100, 500),
        "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 5.0, 15.0)},
    }

    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
    params["kernel"] = kernel

    # gamma meaningful for rbf, poly, sigmoid
    if kernel in ("rbf", "poly", "sigmoid"):
        params["gamma"] = trial.suggest_float("gamma", 1e-6, 10.0, log=True)

    # degree and coef0 only for poly
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
        params["coef0"] = trial.suggest_float("coef0", -5.0, 5.0)

    # coef0 also used by sigmoid
    if kernel == "sigmoid":
        params["coef0"] = trial.suggest_float("coef0_sigmoid", -5.0, 5.0)

    return params


# ── Registry ──────────────────────────────────────────────────────────────────

_SUGGEST_FN = {
    "lgbm": _suggest_lgbm,
    "xgboost": _suggest_xgboost,
    "catboost": _suggest_catboost,
    "hist_gbm": _suggest_hist_gbm,
    "random_forest": _suggest_random_forest,
    "extra_trees": _suggest_extra_trees,
    "gradient_boosting": _suggest_gradient_boosting,
    "lr": _suggest_lr,
    "ridge": _suggest_ridge,
    "svm": _suggest_svm,
}


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for a given model — single entry point.

    All conditionals are resolved here. The returned dict is ready to
    pass directly to the model factory.

    Args:
        trial: Optuna trial.
        model_name: Key matching a registered model (e.g. "lgbm").

    Raises:
        KeyError: If model_name has no registered suggest function.

    Returns:
        Dict of hyperparameters with all conditionals resolved.
    """
    fn = _SUGGEST_FN.get(model_name)
    if fn is None:
        raise KeyError(f"No suggest function for '{model_name}'. Available: {list(_SUGGEST_FN)}")
    return fn(trial)
