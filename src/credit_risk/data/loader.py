from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from credit_risk.config import get_settings


def get_data_path() -> Path:
    """Return the path to the data directory."""
    return get_settings().data_path


def load_application(train: bool = True) -> pl.DataFrame:
    """Load the main application data.

    Args:
        train: If True, load training data; otherwise load test data.

    Returns:
        Polars DataFrame with application data.
    """
    filename = "application_train.csv" if train else "application_test.csv"
    return pl.read_csv(get_data_path() / filename)


def load_previous_application() -> pl.DataFrame:
    """Load previous application data."""
    return pl.read_csv(get_data_path() / "previous_application.csv")


def load_installments_payments() -> pl.DataFrame:
    """Load installments payments data."""
    return pl.read_csv(get_data_path() / "installments_payments.csv")


def load_bureau() -> pl.DataFrame:
    """Load bureau data."""
    return pl.read_csv(get_data_path() / "bureau.csv")


def load_bureau_balance() -> pl.DataFrame:
    """Load bureau balance data."""
    return pl.read_csv(get_data_path() / "bureau_balance.csv")


def load_pos_cash_balance() -> pl.DataFrame:
    """Load POS CASH balance data."""
    return pl.read_csv(get_data_path() / "POS_CASH_balance.csv")


def load_credit_card_balance() -> pl.DataFrame:
    """Load credit card balance data."""
    return pl.read_csv(get_data_path() / "credit_card_balance.csv")


def load_sample_submission() -> pl.DataFrame:
    """Load sample submission format."""
    return pl.read_csv(get_data_path() / "sample_submission.csv")


def load_table(
    name: Literal[
        "application",
        "application_train",
        "application_test",
        "previous_application",
        "installments_payments",
        "bureau",
        "bureau_balance",
        "pos_cash_balance",
        "credit_card_balance",
        "sample_submission",
    ],
) -> pl.DataFrame:
    """Load a table by name.

    Args:
        name: Name of the table to load.

    Returns:
        Polars DataFrame with the requested data.
    """
    loaders = {
        "application": lambda: load_application(train=True),
        "application_train": lambda: load_application(train=True),
        "application_test": lambda: load_application(train=False),
        "previous_application": load_previous_application,
        "installments_payments": load_installments_payments,
        "bureau": load_bureau,
        "bureau_balance": load_bureau_balance,
        "pos_cash_balance": load_pos_cash_balance,
        "credit_card_balance": load_credit_card_balance,
        "sample_submission": load_sample_submission,
    }
    return loaders[name]()


def get_table_info(
    name: Literal[
        "application",
        "application_train",
        "application_test",
        "previous_application",
        "installments_payments",
        "bureau",
        "bureau_balance",
        "pos_cash_balance",
        "credit_card_balance",
        "sample_submission",
    ],
) -> dict:
    """Get basic information about a table.

    Args:
        name: Name of the table.

    Returns:
        Dictionary with shape, columns, and dtypes.
    """
    df = load_table(name)
    return {
        "shape": df.shape,
        "columns": df.columns,
        "dtypes": df.dtypes,
    }
