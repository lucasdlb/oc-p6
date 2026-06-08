"""Credit_card_balance table imputer."""

from __future__ import annotations

from credit_risk_processing.data.imputation.helpers import MedianAndModeImputer


class CreditCardBalanceImputer(MedianAndModeImputer):
    """Imputer for credit_card_balance table.

    Uses median for numeric columns, mode for categorical.
    """
