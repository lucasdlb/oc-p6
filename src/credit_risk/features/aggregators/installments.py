"""Installments payments table aggregator."""

from __future__ import annotations

import polars as pl
from polars import LazyFrame

from credit_risk.config import FeaturesConfig
from credit_risk.features.aggregators.base import TableAggregator

AGG_METHODS = {
    "minimal": ["mean", "max", "count"],
    "default": ["mean", "sum", "min", "max", "std", "count"],
    "detailed": ["mean", "sum", "min", "max", "std", "count", "quantile"],
}


def _get_agg_exprs(
    columns: list[str], numeric_cols: set[str], prefix: str = "", methods: list[str] | None = None
) -> list[pl.Expr]:
    if methods is None:
        methods = list(AGG_METHODS["default"])
    exprs = []
    for col in columns:
        if col in numeric_cols:
            for func in methods:
                if func == "quantile":
                    exprs.append(pl.col(col).quantile(0.5).alias(f"{prefix}{col}_quantile"))
                else:
                    exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
        else:
            for func in ["first", "count"]:
                exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
    return exprs


def _get_numeric_cols(lf: LazyFrame, columns: list[str]) -> set[str]:
    schema = lf.collect_schema()
    return {col for col in columns if col in schema and schema[col] not in (pl.String, pl.Boolean)}


class InstallmentsAggregator(TableAggregator):
    """Aggregator for installments_payments table."""

    def __init__(self, config: FeaturesConfig | None = None):
        from credit_risk.config import cfg

        self.config = config or cfg.data.features

    @classmethod
    def load_link(cls) -> LazyFrame | None:
        """Installments has SK_ID_CURR directly, no link needed."""
        return None

    def aggregate(self, df: LazyFrame, method: str = "default") -> LazyFrame:
        """Aggregate installments_payments to SK_ID_CURR level.

        Args:
            df: installments_payments LazyFrame
            method: Aggregation method ("minimal", "default", "detailed")

        Returns:
            Aggregated LazyFrame with SK_ID_CURR as key
        """
        agg_cols = self.config.installments_agg_features
        numeric_cols = _get_numeric_cols(df, agg_cols)
        methods = AGG_METHODS.get(method, AGG_METHODS["default"])
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "ins_", methods)

        if method != "minimal":
            agg_exprs.append(
                (pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT"))
                .mean()
                .alias("ins_PAYMENT_DIFF_mean")
            )
        if method == "detailed":
            agg_exprs.append(
                ((pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT")) / pl.col("AMT_INSTALMENT"))
                .mean()
                .alias("ins_PAYMENT_RATIO_mean")
            )

        agg_exprs.append(pl.col("SK_ID_CURR").count().alias("ins_n_records"))
        return df.group_by("SK_ID_CURR").agg(*agg_exprs)
