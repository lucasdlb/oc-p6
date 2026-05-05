"""Hyperparameter search spaces for Optuna — binary classification on Home Credit.

Design rules
------------
- Seeds fixed to 42, never tuned.
- Operational params (n_jobs, verbose) set as fixed values per model — only
  for params the model actually accepts, avoiding unknown-kwarg errors.
- All conditionals encoded directly in per-model suggest functions.
- class imbalance: always explicit pos weight bracketing the true ~8% positive
  rate (n_neg/n_pos ≈ 11.5, range 10–13).
- Dataset context: ~82k–197k fold train rows, 195 selected features.
- Early stopping removed from sklearn models — we control iterations directly
  via n_estimators/max_iter to avoid interaction with our outer CV folds.
- SVM and GradientBoosting removed from defaults:
  SVM is O(n²/n³) — impractical at 82k rows.
  GradientBoosting has no class_weight support — unsuitable for 8% imbalance.

Public API
----------
suggest_params(trial, model_name) -> dict
    Single entry point used by the Optuna objective.
"""

from __future__ import annotations

from typing import Any

import optuna

# ── Per-model suggest functions ───────────────────────────────────────────────


def _suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Fixed operational params
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
        # GPU acceleration via OpenCL — max_bin capped at 255 (OpenCL kernel limit)
        "device": "gpu",
        "gpu_use_dp": True,
        # Objective fixed — cross_entropy is an alias; cross_entropy_lambda
        # outputs raw logits which distorts cross-trial ROC AUC comparisons.
        "objective": "binary",
        # boosting_type fixed to gbdt — dart adds stochasticity that
        # destabilises Optuna's surrogate; rf mode is a different model class.
        "boosting_type": "gbdt",
        # Core
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 512),
        # num_leaves constrained to <= 2^max_depth — enforce at fit time in factory
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Dataset-size relative: 20 on 82k = 0.02%, 2000 = 2.4%
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        # Sampling
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # Bagging: always active (bagging_freq=1), tune fraction only
        "bagging_freq": 1,
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # Regularisation
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        # min_data_in_leaf is the native name of min_child_samples — do not set both
        "min_data_in_bin": trial.suggest_int("min_data_in_bin", 3, 100),
        # max_bin capped at 255 — OpenCL GPU kernel hard limit
        "max_bin": trial.suggest_int("max_bin", 63, 255),
        # Imbalance — n_neg/n_pos ≈ 11.5 for 8% positive rate
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10.0, 13.0),
        # Extra trees mode: randomises split points for faster/regularized trees
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
    }
    return params


def _suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Fixed operational params
        "n_jobs": -1,
        "verbosity": 0,
        "random_state": 42,
        # tree_method fixed to hist — fastest, supports both grow policies,
        # default since XGBoost 1.6; "approx" does not support lossguide.
        "tree_method": "hist",
        # Core
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Float: sum of Hessian in leaf — log scale appropriate
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 100.0, log=True),
        # Sampling — colsample_bytree only; bylevel/bynode multiply with it
        # causing effective fractions of ~6% at lower bounds (195 × 0.4³)
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        # Regularisation
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 63, 511),
        # Imbalance
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10.0, 13.0),
    }

    # grow_policy: lossguide = leaf-wise (like LightGBM), depthwise = level-wise
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params["grow_policy"] = grow_policy
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 15, 511)

    return params


def _suggest_catboost(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {
        # Fixed operational params
        "thread_count": -1,
        "verbose": 0,
        "random_seed": 42,
        # boosting_type fixed to Plain — Ordered is significantly slower and
        # benefits mainly small datasets; impractical at 82k+ rows.
        "boosting_type": "Plain",
        # Core
        "iterations": trial.suggest_int("iterations", 500, 5000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 20.0, log=True),
        # Dataset-size relative: 5–500 for 82k–197k fold rows
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 500, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        # Regularisation
        "random_strength": trial.suggest_float("random_strength", 0.01, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        # Leaf estimation
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 20),
        # Missing values
        "nan_mode": trial.suggest_categorical("nan_mode", ["Min", "Max"]),
        # Imbalance
        "class_weights": {0: 1, 1: trial.suggest_float("pos_weight_cat", 10.0, 13.0)},
    }

    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params["grow_policy"] = grow_policy

    # leaf_estimation_method: Exact only valid for SymmetricTree
    if grow_policy == "SymmetricTree":
        params["leaf_estimation_method"] = trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient", "Exact"]
        )
        params["score_function"] = trial.suggest_categorical("score_function", ["Cosine", "L2"])
    elif grow_policy == "Depthwise":
        params["leaf_estimation_method"] = trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        )
        params["score_function"] = trial.suggest_categorical("score_function", ["Cosine", "L2"])
    else:  # Lossguide — score_function must be L2
        params["leaf_estimation_method"] = trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        )
        params["score_function"] = "L2"

    # bootstrap_type drives bagging_temperature vs subsample
    bootstrap = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "No"])
    params["bootstrap_type"] = bootstrap
    if bootstrap == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
    elif bootstrap in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)

    return params


def _suggest_hist_gbm(trial: optuna.Trial) -> dict[str, Any]:
    # No n_jobs, no verbose. Early stopping removed — we control iterations
    # directly via max_iter; internal validation_fraction would train on a
    # subset of the fold, degrading our CV estimates.
    return {
        "random_state": 42,
        "early_stopping": False,
        "max_iter": trial.suggest_int("max_iter", 200, 3000, log=True),
        # max_depth removed — control complexity via max_leaf_nodes only
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 511),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # Dataset-size relative: 10–2000 for 82k–197k fold rows
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 2000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 20.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "max_bins": trial.suggest_int("max_bins", 63, 255),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 10.0, 13.0)},
    }


def _suggest_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    # max_depth removed — control complexity via max_leaf_nodes only.
    # oob_score removed — diagnostic only, no effect on predictions.
    params: dict[str, Any] = {
        "n_jobs": -1,
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        # Dataset-size relative: 5–200 for 82k rows
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 200, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 200, log=True),
        # Upper bound 0.1 — 0.5 would force near-single-split trees
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 511),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 0.1, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 10.0, 13.0)},
        "bootstrap": True,
    }
    return params


def _suggest_extra_trees(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_jobs": -1,
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 200, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 200, log=True),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 511),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 0.1, log=True),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 10.0, 13.0)},
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }


def _suggest_lr(trial: optuna.Trial) -> dict[str, Any]:
    # fit_intercept removed — should always be True for a well-specified model.
    # liblinear removed — O(n×d) per iteration, slow at 82k+ rows; lbfgs/saga preferred.
    params: dict[str, Any] = {
        "n_jobs": -1,
        "random_state": 42,
        "fit_intercept": True,
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 500, 5000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 10.0, 13.0)},
    }

    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    params["solver"] = solver

    if solver == "saga":
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        params["penalty"] = penalty
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:  # lbfgs
        params["penalty"] = "l2"

    return params


def _suggest_ridge(trial: optuna.Trial) -> dict[str, Any]:
    # svd and cholesky removed — O(n³), completely impractical at 82k rows.
    return {
        "random_state": 42,
        "fit_intercept": True,
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "solver": trial.suggest_categorical("solver", ["auto", "lsqr", "sag", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 500, 5000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "class_weight": {0: 1, 1: trial.suggest_float("pos_weight", 10.0, 13.0)},
    }


# ── Registry ──────────────────────────────────────────────────────────────────
# SVM removed: O(n²/n³) — impractical at 82k+ rows.
# GradientBoosting removed: no class_weight support — unsuitable for 8% imbalance.

_SUGGEST_FN = {
    "lgbm": _suggest_lgbm,
    "xgboost": _suggest_xgboost,
    "catboost": _suggest_catboost,
    "hist_gbm": _suggest_hist_gbm,
    "random_forest": _suggest_random_forest,
    "extra_trees": _suggest_extra_trees,
    "lr": _suggest_lr,
    "ridge": _suggest_ridge,
}


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for a given model — single entry point.

    All conditionals and fixed operational constants (n_jobs, verbose,
    random_state, etc.) are included in the returned dict.  Each model
    only receives the params it actually accepts — no unknown-kwarg errors.

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
