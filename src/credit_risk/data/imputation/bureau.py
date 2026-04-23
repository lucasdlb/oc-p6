"""Bureau table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.base import MedianAndModeImputer, TableImputer


class BureauImputer(MedianAndModeImputer, TableImputer):
    """Imputer for bureau table.

    Uses median for numeric columns, mode for categorical.
    """
