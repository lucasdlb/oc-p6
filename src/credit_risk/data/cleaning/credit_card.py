"""Credit card balance table cleaner — domain-aware cleaning."""

from __future__ import annotations

from typing import override

import polars as pl
from polars import DataFrame

from credit_risk.data.base import StatelessStep

_DRAWING_COLS = [
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_OTHER_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
]
_DRAWING_CNT_COLS = [
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_OTHER_CURRENT",
    "CNT_DRAWINGS_POS_CURRENT",
]


class CreditCardBalanceCleaner(StatelessStep):
    """Cleaner for credit_card_balance table.

    Domain-aware cleaning:
    - Null drawing amounts/counts mean "no activity" — fill to 0
    - Null AMT_PAYMENT_CURRENT means "no payment" — fill to 0
    - Null AMT_INST_MIN_REGULARITY means "no minimum required" — fill to 0
    """

    @override
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        drawing_exprs = [pl.col(c).fill_null(0) for c in _DRAWING_COLS]
        cnt_exprs = [pl.col(c).fill_null(0) for c in _DRAWING_CNT_COLS]
        payment_expr = pl.col("AMT_PAYMENT_CURRENT").fill_null(0)
        min_reg_expr = pl.col("AMT_INST_MIN_REGULARITY").fill_null(0)

        all_exprs = drawing_exprs + cnt_exprs + [payment_expr, min_reg_expr]
        return X.with_columns(all_exprs)
