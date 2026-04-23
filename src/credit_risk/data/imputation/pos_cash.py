"""POS_CASH_balance table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.base import MedianAndModeImputer, TableImputer


class POSCashImputer(MedianAndModeImputer, TableImputer):
    """Imputer for POS_CASH_balance table.

    Uses median for numeric columns, mode for categorical.
    """
