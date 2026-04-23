"""Installments_payments table imputer."""

from __future__ import annotations

from credit_risk.data.imputation.base import MedianAndModeImputer, TableImputer


class InstallmentsImputer(MedianAndModeImputer, TableImputer):
    """Imputer for installments_payments table.

    Uses median for numeric columns, mode for categorical.
    """
