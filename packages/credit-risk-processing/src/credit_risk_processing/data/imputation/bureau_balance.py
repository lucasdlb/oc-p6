"""Bureau_balance table imputer."""

from __future__ import annotations

from credit_risk_processing.data.imputation.helpers import DefaultNumericImputer


class BureauBalanceImputer(DefaultNumericImputer):
    """Imputer for bureau_balance table.

    Uses median for numeric columns (MONTHS_BALANCE).
    STATUS is categorical - no imputation needed.
    """
