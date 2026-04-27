"""Installments_payments table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.helpers import MedianAndModeImputer


class InstallmentsImputer(MedianAndModeImputer):
    """Imputer for installments_payments table.

    Uses median for numeric columns, mode for categorical.
    """
