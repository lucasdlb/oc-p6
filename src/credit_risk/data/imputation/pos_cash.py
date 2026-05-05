"""POS_CASH_balance table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.helpers import MedianAndModeImputer


class POSCashBalanceImputer(MedianAndModeImputer):
    """Imputer for POS_CASH_balance table.

    Uses median for numeric columns, mode for categorical.
    """
