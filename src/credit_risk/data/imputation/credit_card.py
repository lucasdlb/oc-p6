"""Credit_card_balance table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.base import MedianAndModeImputer, TableImputer


class CreditCardImputer(MedianAndModeImputer, TableImputer):
    """Imputer for credit_card_balance table.

    Uses median for numeric columns, mode for categorical.
    """
