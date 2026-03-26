"""Feature aggregators for cross-table data using Polars lazy evaluation."""

import polars as pl
from polars import DataFrame, LazyFrame

from credit_risk.config.settings import FeatureConfig

NUMERIC_AGG_FUNCTIONS = ["mean", "sum", "min", "max", "std"]
STRING_AGG_FUNCTIONS = ["first", "count"]


def _get_agg_exprs(columns: list[str], numeric_cols: set[str], prefix: str = "") -> list[pl.Expr]:
    exprs = []
    for col in columns:
        if col in numeric_cols:
            for func in NUMERIC_AGG_FUNCTIONS:
                exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
            exprs.append(pl.col(col).count().alias(f"{prefix}{col}_count"))
        else:
            for func in STRING_AGG_FUNCTIONS:
                exprs.append(getattr(pl.col(col), func)().alias(f"{prefix}{col}_{func}"))
    return exprs


def _get_numeric_cols(lf: LazyFrame, columns: list[str]) -> set[str]:
    schema = lf.collect_schema()
    return {col for col in columns if col in schema and schema[col] not in (pl.String, pl.Boolean)}


class FeatureAggregator:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def aggregate_bureau(self, bureau_df: LazyFrame, id_col: str = "SK_ID_CURR") -> LazyFrame:
        agg_cols = self.config.bureau_agg_features
        numeric_cols = _get_numeric_cols(bureau_df, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "bureau_")
        return bureau_df.group_by(id_col).agg(*agg_exprs)

    def aggregate_bureau_balance(
        self, bureau_balance_df: LazyFrame, bureau_df: LazyFrame, id_col: str = "SK_ID_CURR"
    ) -> LazyFrame:
        bureau_with_id = bureau_df.select("SK_ID_BUREAU", "SK_ID_CURR")
        joined = bureau_balance_df.join(bureau_with_id, on="SK_ID_BUREAU", how="left")
        agg_cols = self.config.bureau_balance_agg_features
        numeric_cols = _get_numeric_cols(joined, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "bb_")
        agg_exprs.append(pl.col("STATUS").mode().first().alias("bb_STATUS_mode"))
        return joined.group_by(id_col).agg(*agg_exprs)

    def aggregate_previous_application(
        self, prev_app_df: LazyFrame, id_col: str = "SK_ID_CURR"
    ) -> LazyFrame:
        agg_cols = self.config.previous_app_agg_features
        numeric_cols = _get_numeric_cols(prev_app_df, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "prev_")
        return prev_app_df.group_by(id_col).agg(*agg_exprs)

    def aggregate_POS_CASH(self, pos_df: LazyFrame, id_col: str = "SK_ID_CURR") -> LazyFrame:
        agg_cols = self.config.pos_cash_agg_features
        numeric_cols = _get_numeric_cols(pos_df, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "pos_")
        return pos_df.group_by(id_col).agg(*agg_exprs)

    def aggregate_installments(
        self, installments_df: LazyFrame, id_col: str = "SK_ID_CURR"
    ) -> LazyFrame:
        agg_cols = self.config.installments_agg_features
        numeric_cols = _get_numeric_cols(installments_df, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "ins_")
        agg_exprs.append(
            (pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT")).mean().alias("ins_PAYMENT_DIFF_mean")
        )
        agg_exprs.append(
            ((pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT")) / pl.col("AMT_INSTALMENT"))
            .mean()
            .alias("ins_PAYMENT_RATIO_mean")
        )
        return installments_df.group_by(id_col).agg(*agg_exprs)

    def aggregate_credit_card(self, cc_df: LazyFrame, id_col: str = "SK_ID_CURR") -> LazyFrame:
        agg_cols = self.config.credit_card_agg_features
        numeric_cols = _get_numeric_cols(cc_df, agg_cols)
        agg_exprs = _get_agg_exprs(agg_cols, numeric_cols, "cc_")
        return cc_df.group_by(id_col).agg(*agg_exprs)

    def aggregate_all(
        self,
        bureau_df: LazyFrame,
        bureau_balance_df: LazyFrame,
        prev_app_df: LazyFrame,
        pos_df: LazyFrame,
        installments_df: LazyFrame,
        cc_df: LazyFrame,
        id_col: str = "SK_ID_CURR",
    ) -> DataFrame:
        bureau_agg = self.aggregate_bureau(bureau_df, id_col).collect()
        bureau_balance_agg = self.aggregate_bureau_balance(
            bureau_balance_df, bureau_df, id_col
        ).collect()
        prev_app_agg = self.aggregate_previous_application(prev_app_df, id_col).collect()
        pos_agg = self.aggregate_POS_CASH(pos_df, id_col).collect()
        installments_agg = self.aggregate_installments(installments_df, id_col).collect()
        cc_agg = self.aggregate_credit_card(cc_df, id_col).collect()

        result = bureau_agg.join(bureau_balance_agg, on=id_col, how="outer_coalesce")  # type: ignore[arg-type]  # outer_coalesce not in ty stubs
        result = result.join(prev_app_agg, on=id_col, how="outer_coalesce")  # type: ignore[arg-type]  # outer_coalesce not in ty stubs
        result = result.join(pos_agg, on=id_col, how="outer_coalesce")  # type: ignore[arg-type]  # outer_coalesce not in ty stubs
        result = result.join(installments_agg, on=id_col, how="outer_coalesce")  # type: ignore[arg-type]  # outer_coalesce not in ty stubs
        result = result.join(cc_agg, on=id_col, how="outer_coalesce")  # type: ignore[arg-type]  # outer_coalesce not in ty stubs

        return result
