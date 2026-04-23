"""Bureau_balance table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.base import DefaultNumericImputer, TableImputer


class BureauBalanceImputer(DefaultNumericImputer, TableImputer):
    """Imputer for bureau_balance table.

    Uses median for numeric columns (MONTHS_BALANCE).
    STATUS is categorical - no imputation needed.
    """
