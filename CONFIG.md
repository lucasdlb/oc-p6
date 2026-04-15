# Configuration Guide

This document explains the configuration system for the credit_risk project.

## Overview

The system uses **two configuration files**:

1. **`data.toml`** - Static data configuration (paths, features, column lists)
2. **`{mode}.toml`** - Runtime configuration (debug/dev/prod modes)

---

## 1. Data Configuration (`configs/data.toml`)

Static configuration that doesn't change between run modes.

### Sections

| Section | Purpose |
|---------|---------|
| `[data]` | Directory paths (raw, processed, output) |
| `[target]` | Target and ID column names |
| `[sources]` | CSV file paths for each data source |
| `[features]` | Columns to drop, categorical columns, EXT_SOURCE |
| `[table.agg]` | Aggregation features per table (bureau, previous_application, etc.) |

### Example

```toml
[data]
raw_dir = "data"
processed_dir = "data/processed"

[target]
column = "TARGET"
id_column = "SK_ID_CURR"

[sources]
application = "data/application_train.csv"
bureau = "data/bureau.csv"

[features]
drop_always = ["SK_ID_CURR", "SK_ID_BUREAU"]
ext_source = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

[bureau.agg]
include = true
features = ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_ANNUITY"]
```

---

## 2. Runtime Configuration (`configs/{mode}.toml`)

Three modes: `debug.toml`, `dev.toml`, `prod.toml`

### Sections

| Section | Purpose | Example |
|---------|---------|---------|
| `[run]` | Execution mode and sampling | `mode = "debug"`, `sample_fraction = 0.05` |
| `[splitter]` | Train/test split settings | `n_splits = 5`, `test_size = 0.2` |
| `[model]` | Model hyperparameters | `max_depth = 3`, `n_estimators = 100` |
| `[selection]` | Feature selection settings | `min_features = 5`, `tolerance = 0.01` |
| `[importance]` | Feature importance strategy | `method = "inner"` |
| `[mlflow]` | MLflow tracking settings | `enabled = true` |
| `[search]` | Grid search preset | `preset = "fast"` |

### Mode Comparison

| Setting | debug | dev | prod |
|---------|-------|-----|------|
| `sample_fraction` | 0.05 (5%) | 0.30 (30%) | 1.0 (100%) |
| `n_splits` | 2 | 3 | 5 |
| `n_estimators` | 50 | 100 | 500 |
| `mlflow.enabled` | false | true | true |

### Example: debug.toml

```toml
[run]
mode = "debug"
sample_fraction = 0.05

[splitter]
test_size = 0.8
n_splits = 2
random_state = 42

[model]
max_depth = 3
n_estimators = 50
learning_rate = 0.05
class_weight = "balanced"

[selection]
min_features = 5
tolerance = 0.01
nb_remove_features = 1

[importance]
method = "inner"

[mlflow]
enabled = false
```

---

## 3. Usage

### Loading Configuration

```python
from credit_risk.config import cfg, load_config

# Loads config based on RUN_MODE env var (debug/dev/prod)
# Default is prod if not set
load_config()

# Access values
print(cfg.run.mode)           # "debug"
print(cfg.model.max_depth)    # 3
print(cfg.importance.method)  # "inner"
```

### Running Scripts

```bash
# Uses debug.toml (5% sample, 2 splits)
RUN_MODE=debug uv run python scripts/rfe_cv.py --table application

# Uses dev.toml (30% sample, 3 splits)
RUN_MODE=dev uv run python scripts/rfe_cv.py --table application

# Uses prod.toml (full data, 5 splits) - default
uv run python scripts/rfe_cv.py --table application
```

---

## 4. Feature Importance Strategies

Configured via `[importance]` section:

```toml
[importance]
method = "inner"  # or "statistical", "permutation", "shap"
```

### Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `"inner"` | Uses model's built-in `feature_importances_` | Default, fast |
| `"statistical"` | Chi2/f_classif scores | Lightweight filtering |
| `"permutation"` | sklearn permutation importance | Model-agnostic |
| `"shap"` | SHAP values with KernelExplainer | Best accuracy, slower |

### Programmatic Usage

```python
from credit_risk.models.importance import get_importance_class

# Get class from config method
ImportanceClass = get_importance_class(cfg.importance.method)
importance_strategy = ImportanceClass()

# Or use directly
from credit_risk.models.importance import (
    InnerImportance,
    StatisticalImportance,
    PermutationImportance,
    SHAPImportance,
)

strategy = SHAPImportance(n_samples=500)
```

---

## 5. Grid Search Presets

Configured via `[search]` section:

```toml
[search]
preset = "fast"
max_grid_values = 3
```

### Available Presets

| Preset | Models | Description |
|--------|--------|-------------|
| `"debug"` | LogisticRegression only | Fastest, 1 param value |
| `"fast"` | LogisticRegression, RandomForest, LightGBM | Quick iteration |
| `"full"` | All three models | Complete search |

### Programmatic Usage

```python
from credit_risk.config.registry import get_configs, trim_grid, PRESETS

# Get configs for preset
configs = get_configs("fast")

# Trim to max N values per parameter (for quick testing)
trimmed = trim_grid(configs, max_values=1)

# List available presets
print(PRESETS.keys())  # dict_keys(['debug', 'fast', 'full'])
```

---

## 6. Configuration Models

Pydantic models in `src/credit_risk/config/models.py`:

| Model | Fields |
|-------|--------|
| `RunConfig` | `mode`, `sample_fraction` |
| `SplitterConfig` | `test_size`, `n_splits`, `random_state` |
| `ModelConfig` | `max_depth`, `n_estimators`, `learning_rate`, etc. |
| `SelectionConfig` | `min_features`, `tolerance`, `nb_remove_features` |
| `ImportanceConfig` | `method` |
| `MLFlowConfig` | `enabled`, `experiment_name` |
| `SearchConfig` | `preset`, `max_grid_values` |
| `DataConfig` | Paths, target, sources, features |
| `Config` | Full config combining all above |

---

## 7. Examples

### Custom Run Mode

Create `configs/custom.toml`:

```toml
[run]
mode = "custom"
sample_fraction = 0.10

[splitter]
n_splits = 3

[model]
max_depth = 4
n_estimators = 200
learning_rate = 0.03

[importance]
method = "shap"

[mlflow]
enabled = true
experiment_name = "custom_experiment"
```

Run with:
```bash
RUN_MODE=custom uv run python scripts/rfe_cv.py --table application
```

### Change Importance Method

In any mode config:

```toml
[importance]
method = "shap"  # Use SHAP importance
```

### Grid Search

```toml
[search]
preset = "full"      # Use all models
max_grid_values = 2  # Keep 2 values per param
```

---

## 8. File Structure

```
configs/
├── data.toml        # Static data config (loaded always)
├── debug.toml       # Debug mode (5% data, 2 splits)
├── dev.toml         # Dev mode (30% data, 3 splits)
└── prod.toml        # Prod mode (full data, 5 splits)

src/credit_risk/config/
├── __init__.py      # Exports: cfg, load_config, Config, get_configs
├── config.py        # load_config(), global cfg
├── models.py        # Pydantic models
├── registry.py      # ModelGridConfig, PRESETS, get_configs()
└── settings.py      # Legacy settings (deprecated)
```

---

## 9. Anti-Patterns to Avoid

1. **Don't hardcode paths** - Use `cfg.data.raw_dir`
2. **Don't create new Config instances** - Use `load_config()` singleton
3. **Don't skip validation** - TOML is validated by Pydantic
4. **Don't forget RUN_MODE** - Defaults to prod if not set
