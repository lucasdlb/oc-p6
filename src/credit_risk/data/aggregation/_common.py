"""Shared aggregation utilities extracted from table-specific aggregators."""

from __future__ import annotations

import polars as pl
from polars import LazyFrame

NUMERIC_AGG_FUNCTIONS = ["mean", "sum", "min", "max", "std"]

AGG_METHODS = {
    "minimal": ["mean", "max", "count"],
    "default": ["mean", "sum", "min", "max", "std", "count"],
    "detailed": ["mean", "sum", "min", "max", "std", "count", "quantile"],
}


def _is_string_col(col: str, schema: dict[str, pl.DataType]) -> bool:
    return col in schema and schema[col] == pl.String


def _recency_weighted_mean(
    col: str,
    months_col: str = "MONTHS_BALANCE",
    half_life: float = 6.0,
) -> pl.Expr:
    """Exponentially decay-weighted mean.

    Recent months count more than distant months.
    weight = exp(-0.693 * |months| / half_life) so that at distance = half_life,
    weight = 0.5 (half weight of most recent month).

    Parameters
    ----------
    col : column to aggregate
    months_col : time column (negative values, most recent = largest value)
    half_life : months at which weight drops to 0.5
    """
    weight = (-0.693 / half_life * pl.col(months_col).abs()).exp()
    return (pl.col(col) * weight).sum() / weight.sum()


def _get_agg_exprs(
    columns: list[str],
    schema: dict[str, pl.DataType],
    prefix: str = "",
    numeric_methods: list[str] | None = None,
) -> list[pl.Expr]:
    """Build aggregation expressions for all columns in a table.

    Automatically handles column types:
    - Numeric (non-String): apply numeric_methods (mean, sum, min, max, std, ...)
    - String (categorical): always apply n_unique + mode + null_count
    - Boolean: same as numeric (sum=count True, mean=proportion True, etc.)

    Parameters
    ----------
    columns : list of column names to aggregate
    schema : {col_name: Polars dtype} from the input DataFrame schema
    prefix : string prefix for all output column names
    numeric_methods : list of aggregation method names for numeric columns
                      (default: AGG_METHODS["default"])
    """
    if numeric_methods is None:
        numeric_methods = list(AGG_METHODS["default"])

    exprs = []
    for col in columns:
        if col not in schema:
            continue

        col_dtype = schema[col]

        if col_dtype in (pl.String, pl.Categorical, pl.Utf8):
            exprs.append(pl.col(col).n_unique().alias(f"{prefix}{col}_n_unique"))
            exprs.append(pl.col(col).mode().first().alias(f"{prefix}{col}_mode"))
            exprs.append(pl.col(col).null_count().alias(f"{prefix}{col}_null_count"))

        else:
            for func in numeric_methods:
                if func == "quantile":
                    exprs.append(pl.col(col).quantile(0.5).alias(f"{prefix}{col}_quantile"))
                else:
                    exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))

    return exprs


def _categorical_diversity_exprs(
    lf: LazyFrame,
    group_col: str,
    cat_col: str,
    prefix: str = "",
) -> LazyFrame:
    """Compute top_freq and entropy for a categorical column per group.

    Returns a LazyFrame with two columns per category:
    - {prefix}{cat_col}_top_freq: fraction of rows in most common category
    - {prefix}{cat_col}_entropy: Shannon entropy of category distribution

    Parameters
    ----------
    lf : LazyFrame with raw table data
    group_col : column to group by (e.g., "SK_ID_CURR")
    cat_col : categorical column to compute diversity metrics for
    prefix : prefix for output column names
    """
    counts = lf.group_by([group_col, cat_col]).agg(pl.len().alias("_count"))

    total_per_group = lf.group_by(group_col).agg(pl.len().alias("_total"))

    counts_with_total = counts.join(total_per_group, on=group_col).with_columns(
        (pl.col("_count") / pl.col("_total")).alias("_p")
    )

    entropy_expr = (
        -(pl.col("_p") * pl.col("_p").log(base=2)).sum().alias(f"{prefix}{cat_col}_entropy")
    )

    top_freq_expr = (pl.col("_count").max() / pl.col("_total").first()).alias(
        f"{prefix}{cat_col}_top_freq"
    )

    return counts_with_total.group_by(group_col).agg(top_freq_expr, entropy_expr)
