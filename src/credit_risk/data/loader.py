"""Data loader module with pandas, polars eager, and polars lazy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

from credit_risk.config import get_settings

TABLE_NAMES = Literal[
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "credit_card_balance",
    "installments_payments",
    "sample_submission",
]

TABLES: dict[str, str] = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "sample_submission": "sample_submission.csv",
}


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    Provides common path resolution and table registry.
    Subclasses must implement ``_read_csv`` to return their preferred frame type.
    """

    def __init__(self, data_path: Path | None = None) -> None:
        self._data_path = data_path or get_settings().data_path

    @property
    def data_path(self) -> Path:
        """Root directory containing CSV files."""
        return self._data_path

    @staticmethod
    def available_tables() -> list[str]:
        """Return the list of supported table names."""
        return list(TABLES)

    def table_path(self, name: str) -> Path:
        """Resolve the CSV path for a given table name.

        Args:
            name: Logical table name.

        Returns:
            Absolute path to the CSV file.

        Raises:
            ValueError: If the table name is unknown.
        """
        if name not in TABLES:
            msg = f"Unknown table {name!r}. Available: {list(TABLES)}"
            raise ValueError(msg)
        return self._data_path / TABLES[name]

    @abstractmethod
    def _read_csv(self, path: Path) -> Any:
        """Read a CSV file and return a dataframe.

        Args:
            path: Path to the CSV file.

        Returns:
            A dataframe in the subclass-specific type.
        """

    def load(self, name: str) -> Any:
        """Load a table by name.

        Args:
            name: Logical table name (see ``available_tables()``).

        Returns:
            A dataframe in the subclass-specific type.
        """
        return self._read_csv(self.table_path(name))

    def load_application_train(self) -> Any:
        """Load the main application training data (TARGET column present)."""
        return self.load("application_train")

    def load_application_test(self) -> Any:
        """Load the main application test data."""
        return self.load("application_test")

    def load_previous_application(self) -> Any:
        """Load previous Home Credit applications."""
        return self.load("previous_application")

    def load_installments_payments(self) -> Any:
        """Load installment payment history."""
        return self.load("installments_payments")

    def load_bureau(self) -> Any:
        """Load bureau credit data."""
        return self.load("bureau")

    def load_bureau_balance(self) -> Any:
        """Load bureau monthly balances."""
        return self.load("bureau_balance")

    def load_pos_cash_balance(self) -> Any:
        """Load POS and cash loan balances."""
        return self.load("pos_cash_balance")

    def load_credit_card_balance(self) -> Any:
        """Load credit card monthly balances."""
        return self.load("credit_card_balance")

    def load_sample_submission(self) -> Any:
        """Load sample submission format."""
        return self.load("sample_submission")


class PLDataLoader(BaseDataLoader):
    """Polars eager data loader.

    Uses ``pl.read_csv`` to return fully materialised ``pl.DataFrame`` objects.
    """

    def _read_csv(self, path: Path) -> pl.DataFrame:
        import polars as pl

        return pl.read_csv(path)


class PLLazyDataLoader(BaseDataLoader):
    """Polars lazy data loader.

    Uses ``pl.scan_csv`` to return ``pl.LazyFrame`` objects.
    Call ``.collect()`` to materialise when ready.
    """

    def _read_csv(self, path: Path) -> pl.LazyFrame:
        import polars as pl

        return pl.scan_csv(path)


class PDDataLoader(BaseDataLoader):
    """Pandas data loader.

    Uses ``pd.read_csv`` to return ``pd.DataFrame`` objects.
    """

    def _read_csv(self, path: Path) -> pd.DataFrame:
        import pandas as pd

        return pd.read_csv(path)


class DataLoader:
    """Convenience loader wrapping ``PLDataLoader``.

    Default: Polars eager (main table) + Polars lazy (auxiliary tables).
    """

    def __init__(self) -> None:
        self._eager = PLDataLoader()
        self._lazy = PLLazyDataLoader()

    def load(self, name: str) -> pl.DataFrame:
        return self._eager.load(name)

    def load_application_train(self) -> pl.DataFrame:
        return self._eager.load_application_train()

    def load_application_test(self) -> pl.DataFrame:
        return self._eager.load_application_test()

    def load_bureau(self) -> pl.LazyFrame:
        return self._lazy.load("bureau")

    def load_bureau_balance(self) -> pl.LazyFrame:
        return self._lazy.load("bureau_balance")

    def load_previous_application(self) -> pl.LazyFrame:
        return self._lazy.load("previous_application")

    def load_pos_cash_balance(self) -> pl.LazyFrame:
        return self._lazy.load("pos_cash_balance")

    def load_credit_card_balance(self) -> pl.LazyFrame:
        return self._lazy.load("credit_card_balance")

    def load_installments_payments(self) -> pl.LazyFrame:
        return self._lazy.load("installments_payments")

    def load_sample_submission(self) -> pl.DataFrame:
        return self._eager.load_sample_submission()

    def load_all_lazy(self) -> dict[str, pl.LazyFrame]:
        return {
            "bureau": self.load_bureau(),
            "bureau_balance": self.load_bureau_balance(),
            "previous_application": self.load_previous_application(),
            "POS_CASH_balance": self.load_pos_cash_balance(),
            "credit_card_balance": self.load_credit_card_balance(),
            "installments_payments": self.load_installments_payments(),
        }
