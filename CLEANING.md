# Data Cleaning Architecture

> `credit_risk` · R&D Guide — how to manage cleaners, experiment with strategies, and ship to production.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Cleaning Methods](#2-cleaning-methods)
3. [The Registry](#3-the-registry)
4. [Configuring Experiments](#4-configuring-experiments)
5. [Writing a Custom Cleaner](#5-writing-a-custom-cleaner)
6. [R&D Patterns](#6-rd-patterns)
7. [Production Checklist](#7-production-checklist)

---

## 1. Overview

The cleaning layer has three collaborating components:

| Component | Responsibility | Entry point |
|---|---|---|
| `CleaningRegistry` | Holds `table → cleaner class` mapping | `CleaningRegistry.get_cleaner(table, method)` |
| `DataCleaner` | Composite facade; reads `DataSourcesConfig` | `cleaner.clean(df, table)` |
| `DataSourcesConfig` | Declares cleaning method per table per experiment | `cleaning_method="default" \| "raw"` |

The call chain for every table:

```
DataCleaner.clean(df, table)
  → _get_cleaning_method(table)       # reads DataSourcesConfig
  → CleaningRegistry.get_cleaner(table, method)
  → TableCleaner.clean(df)            # table-specific logic
```

---

## 2. Cleaning Methods

Two methods are supported at the `DataSourcesConfig` level:

| method | Behaviour | When to use |
|---|---|---|
| `"default"` | Runs the registered `TableCleaner` for this table | Normal training / production |
| `"raw"` | Returns the dataframe unchanged (`RawCleaner`) | Ablation — disable cleaning for one table |

If no `DataSourcesConfig` is passed to `DataCleaner`, all tables fall back to `"default"`.

---

## 3. The Registry

### Default registrations

| Table | Default cleaner |
|---|---|
| `application` | `ApplicationCleaner` |
| `bureau` | `BureauCleaner` |
| `bureau_balance` | `BureauBalanceCleaner` |
| `previous_application` | `PreviousApplicationCleaner` |
| `pos_cash` | `POSCashCleaner` |
| `installments` | `InstallmentsCleaner` |
| `credit_card` | `CreditCardCleaner` |

### Key behaviours

- **Lazy init** — defaults are registered on first `get_cleaner` call, not at import time (avoids circular imports).
- **Hard failure on unknown table** — passing an unregistered table name raises `KeyError` with a list of valid names. There is no silent fallback.
- **`"raw"` short-circuits the registry** — `RawCleaner` is returned immediately, no lookup needed.

### Runtime registration

You can override or add a cleaner at runtime without touching the registry source:

```python
from credit_risk.data.cleaning.registry import CleaningRegistry
from my_experiment.cleaners import MyBureauBalanceCleaner

CleaningRegistry.register("bureau_balance", MyBureauBalanceCleaner)

---

## 4. Configuring Experiments

Cleaning strategy is declared in `DataSourcesConfig`, not in code. This keeps experiment config serializable and reproducible.

### Use default cleaning everywhere (baseline)

```python
from credit_risk.config.experiment_config import DataSourcesConfig

ds = DataSourcesConfig(
    application=True,
    bureau=True,
    bureau_balance=True,
)
# DataCleaner will use "default" for all tables
```

### Disable cleaning for one table (ablation)

```python
from credit_risk.config.experiment_config import DataSourcesConfig, DataSourceSpec

ds = DataSourcesConfig(
    application=True,
    bureau=True,
    bureau_balance=DataSourceSpec(
        enabled=True,
        cleaning_method="raw",   # skip BureauBalanceCleaner
    ),
)
```

### Grid search over cleaning strategies

```python
ds = DataSourcesConfig(
    bureau_balance=[
        DataSourceSpec(enabled=True, cleaning_method="default"),
        DataSourceSpec(enabled=True, cleaning_method="raw"),
    ]
)
# Expands to 2 experiments automatically
```

---

## 5. Writing a Custom Cleaner

All cleaners inherit from `TableCleaner`:

```python
# credit_risk/data/cleaning/base.py
class TableCleaner:
    def clean(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError
```

### Minimal example — `BureauBalanceCleaner`

```python
from polars import DataFrame, col, when
from credit_risk.data.cleaning.base import TableCleaner

STATUS_MAP = {"C": -1, "X": -2}

class BureauBalanceCleaner(TableCleaner):
    def clean(self, df: DataFrame) -> DataFrame:
        return (
            df
            .with_columns(
                col("STATUS")
                .replace(STATUS_MAP)
                .cast(int)
                .alias("STATUS")
            )
            .drop_nulls(subset=["SK_ID_BUREAU"])
        )
```

### fit / transform split (required for production)

If your cleaner learns anything from data (mean, mode, threshold, encoder), split it explicitly:

```python
class BureauBalanceCleaner(TableCleaner):
    def fit(self, df: DataFrame) -> "BureauBalanceCleaner":
        self._status_counts = df["STATUS"].value_counts()
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        # uses only self._status_counts, no df-level statistics
        ...

    def clean(self, df: DataFrame) -> DataFrame:
        # convenience for R&D — fit+transform in one shot
        return self.fit(df).transform(df)
```

> At serving time, only `transform` is called with the frozen fitted state. This boundary must be explicit before productionizing.

---

## 6. R&D Patterns

### Temporary override in a notebook or test

Use the `override` context manager to swap a cleaner for a single run without side effects:

```python
with CleaningRegistry.override("bureau_balance", ExperimentalCleaner):
    pipeline.run(config)
# registry is restored here — other experiments are unaffected
```

### Inspect registered tables

```python
CleaningRegistry.available_tables()
# ["application", "bureau", "bureau_balance", ...]
```

### Typical ablation matrix

| Experiment | `bureau_balance` method | `bureau` method | Goal |
|---|---|---|---|
| baseline | `default` | `default` | Reference score |
| no_bb_clean | `raw` | `default` | Is `BureauBalanceCleaner` helping? |
| no_bureau_clean | `default` | `raw` | Is `BureauCleaner` helping? |
| all_raw | `raw` | `raw` | Lower bound — pure data |

Encode this as a list of `DataSourcesConfig` objects and run them in a loop. Each produces a distinct `run_id` for comparison.

---

## 7. Production Checklist

Before deploying:

- [ ] Every `TableCleaner` has separate `fit` and `transform` methods
- [ ] Fitted cleaner state is serialized alongside the model artifact (`cleaner.pkl` next to `model.pkl`)
- [ ] At inference time only `transform` is called — never `fit`
- [ ] The exact `DataSourcesConfig` used for training is frozen in `config.json` next to the model
- [ ] `CleaningRegistry.register` calls (if any) happen before pipeline init, not inside it
- [ ] No cleaner reads from disk or network inside `clean` — all paths go through `AppSettings`

---

## File Map

```
src/credit_risk/data/cleaning/
├── base.py                    # TableCleaner ABC
├── raw.py                     # RawCleaner (no-op)
├── registry.py                # CleaningRegistry
├── application.py
├── bureau.py
├── bureau_balance.py
├── previous_application.py
├── pos_cash.py
├── installments.py
└── credit_card.py
```
