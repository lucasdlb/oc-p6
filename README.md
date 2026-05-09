# Credit Risk ML Pipeline

Binary classification on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset.
Predicts probability of loan default (`TARGET=1`) from 7 joined tables.
Metric: **ROC-AUC** (stratified, class-weighted training).

---

## Project structure

```
src/credit_risk/
  config/       # Pydantic config models + TOML loader
  data/         # Cleaning → Imputation → Aggregation → Transformation → Encoding
  models/       # CV, feature selection, tuning, importance
  pipeline/     # TableTransformer, ProcessingCV, evaluator
  interpret/    # SHAP explainability
scripts/
  sweep_processing.py   # Grid-search processing hyperparams per table
  rfe_cv.py             # Backward feature elimination (leak-free CV)
  tune.py               # Optuna hyperparameter tuning
  final_train.py        # Final fit on full train set, evaluate on held-out test
configs/
  data.toml             # Static: file paths, table steps (cleaner/imputer/…)
  configs/data/test.toml# Same structure, all real steps — used by pytest
  debug.toml            # 5% sample, 2-fold, 2 trials
  dev.toml              # 50% sample, 3-fold, 30 trials
  prod.toml             # 100% sample, 5-fold, 60 trials
```

---

## Pipeline

### Processing steps (per table)

Each sub-table goes through a fixed chain configured in `data.toml`:

```
Clean → Impute → Aggregate → Transform → Encode
```

All steps implement `fit_transform`. `NoOpStep` is the standard disable
mechanism — swap any step to `"NoOpStep"` in the config to skip it.

### Cross-table features

After per-table processing, `CrossTableTransformer` computes interaction
features across joined tables (e.g. `cross_bureau_bb_active_ever_bad`,
`cross_bureau_overdue_x_delinquency`).

### Leak-free cross-validation

`ProcessingCV` builds fold arrays once (`build_folds`), then each fold:
1. Processes train split through `TableTransformer` (fit on train only)
2. Applies fitted transformers to val split
3. Trains model, scores on val

The test set is locked away until `final_train.py`.

### Feature selection

`BackwardFeatureSelector` runs backward elimination over the joined feature
matrix. At each step the bottom-20% (by importance) are dropped, and CV
AUC is re-evaluated. Selection stops when AUC drops more than `tolerance=0.005`.
Importances are aligned by feature name across folds to handle encoder
boundary differences between folds.

### Tuning

`ProcessingTuner` wraps Optuna. Fold arrays are pre-built once; each trial
is numpy-only (no table reprocessing per trial). GPU acceleration:

| Model | Device | Optuna `n_jobs` |
|---|---|---|
| LightGBM | GPU (OpenCL) | 4 |
| XGBoost | CUDA | 4 |
| CatBoost | GPU (~5.4 GB) | 1 |
| HistGBM / RF / ET | CPU | -1 |

### MLflow tracking

Every script logs to a local SQLite MLflow store (`mlflow.db`).
`features.json` artifact stores the selected feature list per run.
`best_config.json` stores best Optuna trial params.
`final_train.py` resolves both artifacts with a mode fallback: `prod → dev → debug`.

---

## First results (dev mode — 50% sample, 3-fold, LightGBM)

| Stage | Features | CV ROC-AUC |
|---|---|---|
| All features (post-processing) | 737 | 0.7824 |
| After backward RFE | 303 | **0.7829 ± 0.0035** |

Selection eliminated ~59% of features with no AUC loss.

---

## Setup

```bash
uv sync --dev
```

Data files (not committed) go in `data/`:
```
data/application_train.csv
data/bureau.csv
data/bureau_balance.csv
data/previous_application.csv
data/POS_CASH_balance.csv
data/installments_payments.csv
data/credit_card_balance.csv
```

---

## Running the pipeline

```bash
# 1. Sweep processing hyperparams (optional)
RUN_MODE=dev uv run python scripts/sweep_processing.py

# 2. Backward feature selection
RUN_MODE=dev uv run python scripts/rfe_cv.py

# 3. Hyperparameter tuning
RUN_MODE=dev uv run python scripts/tune.py

# 4. Final train + test evaluation
RUN_MODE=dev uv run python scripts/final_train.py
```

Set `RUN_MODE=prod` for full-data runs.

---

## Tests

```bash
uv run pytest                        # all tests (uses configs/data/test.toml)
uv run pytest tests/test_data.py     # data pipeline tests only
uv run pytest --cov=src              # with coverage
```

Tests always use `configs/data/test.toml` (all real steps, no `NoOpStep`
where a real implementation exists) regardless of what `data.toml` says.
Override per-test via `monkeypatch` or `DATA_CONFIG=data` env var.
