"""Data loader module with polars eager and lazy implementations.

All loaders accept ``data_path`` and ``csv_names`` as constructor arguments
— no implicit config loading. The caller must supply the CSV name mapping.
"""

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
    "installments",
    "sample_submission",
]

KNOWN_SCHEMA_OVERRIDES: dict[str, dict[str, type[pl.DataType]]] = {
    "bureau": {
        "AMT_ANNUITY": pl.Float64,
    },
}

TABLE_LOAD_METHODS: dict[str, str] = {
    "bureau_balance": "load_bureau_balance",
    "pos_cash_balance": "load_pos_cash_balance",
    "credit_card_balance": "load_credit_card_balance",
    "installments": "load_installments",
}


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    Provides common path resolution, table registry, and link-joining logic.
    Daughter classes implement _read_csv and _join for their specific framework.

    Args:
        data_path: Root directory containing CSV files.
        csv_names: Mapping of table name → CSV filename.
    """

    def __init__(self, data_path: Path, csv_names: dict[str, str]) -> None:
        self._data_path = data_path
        self._csv_names = csv_names

    @property
    def data_path(self) -> Path:
        """Root directory containing CSV files."""
        return self._data_path

    def available_tables(self) -> list[str]:
        """Return the list of supported table names."""
        return list(self._csv_names)

    def table_path(self, name: str) -> Path:
        """Resolve the CSV path for a given table name."""
        if name not in self._csv_names:
            msg = f"Unknown table {name!r}. Available: {list(self._csv_names)}"
            raise ValueError(msg)
        return self._data_path / self._csv_names[name]

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
            logger.debug("Dispatching '%s' to %s", name, method_name)
            return getattr(self, method_name)()
        path = self.table_path(name)
        known_overrides = KNOWN_SCHEMA_OVERRIDES.get(name)
        if known_overrides:
            logger.debug("Loading '%s' with schema overrides: %s", name, list(known_overrides))
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

    def load_installments(self) -> Any:
        """Load installments with SK_ID_CURR via previous_application link."""
        ins = self._read_csv(self.table_path("installments"))
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
