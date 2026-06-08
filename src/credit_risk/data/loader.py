"""Config-aware loader wrappers — thin layer over credit_risk_data.

Imports the core loader classes from the standalone ``credit_risk_data`` package
and adds config-based path + CSV-name resolution for internal ``oc-p6`` scripts.

External consumers should use ``credit_risk_data`` directly.
"""

from __future__ import annotations

from pathlib import Path

from credit_risk_data.loader import (  # noqa: F401  — re-exported for backward compat
    KNOWN_SCHEMA_OVERRIDES,
    TABLE_LOAD_METHODS,
    TABLE_NAMES,
    BaseDataLoader,
)
from credit_risk_data.loader import (
    PDDataLoader as _PDDataLoader,
)
from credit_risk_data.loader import (
    PLDataLoader as _PLDataLoader,
)
from credit_risk_data.loader import (
    PLLazyDataLoader as _PLLazyDataLoader,
)


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
        "pos_cash_balance": cfg.data.sources.pos_cash_balance,
        "credit_card_balance": cfg.data.sources.credit_card_balance,
        "installments": cfg.data.sources.installments,
        "sample_submission": "sample_submission.csv",
    }


def _resolve(data_path: Path | None) -> tuple[Path, dict[str, str]]:
    """Resolve data_path and csv_names from config."""
    from credit_risk.config import load_config
    from credit_risk.config.config import PROJECT_ROOT

    cfg = load_config()
    csv_names = get_table_csv_names()
    path = data_path or PROJECT_ROOT / cfg.data.data_dir
    return path, csv_names


class PLDataLoader(_PLDataLoader):
    """Polars eager loader with config-backed defaults."""

    def __init__(self, data_path: Path | None = None) -> None:
        path, csv_names = _resolve(data_path)
        super().__init__(data_path=path, csv_names=csv_names)


class PLLazyDataLoader(_PLLazyDataLoader):
    """Polars lazy loader with config-backed defaults."""

    def __init__(self, data_path: Path | None = None) -> None:
        path, csv_names = _resolve(data_path)
        super().__init__(data_path=path, csv_names=csv_names)


class PDDataLoader(_PDDataLoader):
    """Pandas loader with config-backed defaults."""

    def __init__(self, data_path: Path | None = None) -> None:
        path, csv_names = _resolve(data_path)
        super().__init__(data_path=path, csv_names=csv_names)
