"""Bureau table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.helpers import MedianAndModeImputer


class BureauImputer(MedianAndModeImputer):
    """Imputer for bureau table.

    Uses median for numeric columns, mode for categorical.
    """
