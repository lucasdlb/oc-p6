"""Previous_application table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.helpers import MedianAndModeImputer


class PreviousApplicationImputer(MedianAndModeImputer):
    """Imputer for previous_application table.

    Uses median for numeric columns, mode for categorical.
    """
