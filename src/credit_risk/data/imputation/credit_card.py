"""Credit_card_balance table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.helpers import MedianAndModeImputer


class CreditCardImputer(MedianAndModeImputer):
    """Imputer for credit_card_balance table.

    Uses median for numeric columns, mode for categorical.
    """
