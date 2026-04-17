"""Data loader module with polars eager and lazy implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

TABLE_NAMES = Literal[
    "application",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "credit_card_balance",
    "installments_payments",
    "sample_submission",
]


def get_table_csv_names() -> dict[str, str]:
    """Get table CSV names from config (lazy, not at import time)."""
    from credit_risk.config import load_config

    cfg = load_config()
    return {
        "application": cfg.data.sources.application,
        "application_test": "application_test.csv",
        "bureau": cfg.data.sources.bureau,
        "bureau_balance": cfg.data.sources.bureau_balance,
        "previous_application": cfg.data.sources.previous_application,
        "pos_cash_balance": cfg.data.sources.pos_cash,
        "credit_card_balance": cfg.data.sources.credit_card,
        "installments_payments": cfg.data.sources.installments,
        "sample_submission": "sample_submission.csv",
    }


# Populated lazily on first access
_tables_csv_names: dict[str, str] | None = None


def _get_tables_csv_names() -> dict[str, str]:
    global _tables_csv_names
    if _tables_csv_names is None:
        _tables_csv_names = get_table_csv_names()
    return _tables_csv_names


# Known schema overrides per table — from audit script.
KNOWN_SCHEMA_OVERRIDES: dict[str, dict[str, type[pl.DataType]]] = {
    "bureau": {
        "AMT_ANNUITY": pl.Float64,
    },
}

# Registry: table name -> specialized load method name
# Used by load() to dispatch to methods that join with link tables to get SK_ID_CURR
TABLE_LOAD_METHODS: dict[str, str] = {
    "bureau_balance": "load_bureau_balance",
    "pos_cash_balance": "load_pos_cash_balance",
    "credit_card_balance": "load_credit_card_balance",
    "installments_payments": "load_installments_payments",
}


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    Provides common path resolution, table registry, and link-joining logic.
    Daughter classes implement _read_csv and _join for their specific framework.
    """

    def __init__(self, data_path: Path | None = None) -> None:
        from credit_risk.config import load_config

        cfg = load_config()

        from credit_risk.config.config import PROJECT_ROOT

        default_path = PROJECT_ROOT / cfg.data.data_dir
        self._data_path = data_path or default_path

    @property
    def data_path(self) -> Path:
        """Root directory containing CSV files."""
        return self._data_path

    @staticmethod
    def available_tables() -> list[str]:
        """Return the list of supported table names."""
        return list(_get_tables_csv_names())

    def table_path(self, name: str) -> Path:
        """Resolve the CSV path for a given table name."""
        tables = _get_tables_csv_names()
        if name not in tables:
            msg = f"Unknown table {name!r}. Available: {list(tables)}"
            raise ValueError(msg)
        return self._data_path / tables[name]

    @abstractmethod
    def _read_csv(
        self, path: Path, schema_overrides: dict[str, type[pl.DataType]] | None = None
    ) -> Any:
        """Read a CSV file and return a dataframe."""

    @abstractmethod
    def _join(self, left: Any, right: Any, on: str) -> Any:
        """Join two dataframes. Override in subclass for framework-specific behavior."""

    def load(self, name: str) -> Any:
        """Load a table by name, dispatching to specialized methods if needed.

        Tables that require SK_ID_CURR join are loaded via their specialized method.
        """
        if name in TABLE_LOAD_METHODS:
            method_name = TABLE_LOAD_METHODS[name]
            logger.debug(f"Dispatching '{name}' to {method_name}")
            return getattr(self, method_name)()
        path = self.table_path(name)
        known_overrides = KNOWN_SCHEMA_OVERRIDES.get(name)
        if known_overrides:
            logger.debug(f"Loading '{name}' with schema overrides: {list(known_overrides)}")
            return self._read_csv(path, schema_overrides=known_overrides)
        return self._read_csv(path)

    def load_bureau_balance(self) -> Any:
        """Load bureau_balance with SK_ID_CURR via bureau link."""
        bb = self._read_csv(self.table_path("bureau_balance"))
        link = self._read_csv(self.table_path("bureau")).select(["SK_ID_BUREAU", "SK_ID_CURR"])
        return self._join(bb, link, "SK_ID_BUREAU")

    def load_pos_cash_balance(self) -> Any:
        """Load POS_CASH_balance with SK_ID_CURR via previous_application link."""
        pos = self._read_csv(self.table_path("pos_cash_balance"))
        link = self._read_csv(self.table_path("previous_application")).select(
            ["SK_ID_PREV", "SK_ID_CURR"]
        )
        return self._join(pos, link, "SK_ID_PREV")

    def load_credit_card_balance(self) -> Any:
        """Load credit_card_balance with SK_ID_CURR via previous_application link."""
        cc = self._read_csv(self.table_path("credit_card_balance"))
        link = self._read_csv(self.table_path("previous_application")).select(
            ["SK_ID_PREV", "SK_ID_CURR"]
        )
        return self._join(cc, link, "SK_ID_PREV")

    def load_installments_payments(self) -> Any:
        """Load installments_payments with SK_ID_CURR via previous_application link."""
        ins = self._read_csv(self.table_path("installments_payments"))
        link = self._read_csv(self.table_path("previous_application")).select(
            ["SK_ID_PREV", "SK_ID_CURR"]
        )
        return self._join(ins, link, "SK_ID_PREV")

    def load_application_train(self) -> Any:
        return self.load("application_train")

    def load_application_test(self) -> Any:
        return self.load("application_test")

    def load_bureau(self) -> Any:
        return self.load("bureau")

    def load_previous_application(self) -> Any:
        return self.load("previous_application")

    def load_sample_submission(self) -> Any:
        return self.load("sample_submission")

    def load_labels(self) -> pl.LazyFrame:
        """Load SK_ID_CURR and TARGET columns as lazy frame."""
        return self.load("application").select(["SK_ID_CURR", "TARGET"]).lazy()


class PLDataLoader(BaseDataLoader):
    """Polars eager data loader."""

    def _read_csv(
        self, path: Path, schema_overrides: dict[str, pl.DataType] | None = None
    ) -> pl.DataFrame:
        return pl.read_csv(path, schema_overrides=schema_overrides or {})

    def _join(
        self, left: pl.LazyFrame | pl.DataFrame, right: pl.LazyFrame | pl.DataFrame, on: str
    ) -> pl.LazyFrame | pl.DataFrame:
        return left.join(right, on=on, how="left")


class PLLazyDataLoader(BaseDataLoader):
    """Polars lazy data loader."""

    def _read_csv(
        self, path: Path, schema_overrides: dict[str, pl.DataType] | None = None
    ) -> pl.LazyFrame:
        return pl.scan_csv(path, schema_overrides=schema_overrides or {})

    def _join(
        self, left: pl.LazyFrame | pl.DataFrame, right: pl.LazyFrame | pl.DataFrame, on: str
    ) -> pl.LazyFrame:
        return left.join(right, on=on, how="left")


class PDDataLoader(BaseDataLoader):
    """Pandas data loader."""

    def _read_csv(
        self, path: Path, schema_overrides: dict[str, pl.DataType] | None = None
    ) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_csv(path)

    def _join(self, left: "pd.DataFrame", right: "pd.DataFrame", on: str) -> "pd.DataFrame":
        return left.merge(right, on=on, how="left")
