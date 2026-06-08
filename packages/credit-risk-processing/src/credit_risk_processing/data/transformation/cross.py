"""Cross-table feature engineering — computed from the merged post-join DataFrame.

CrossTableTransformer is applied by TableTransformer after all per-table
pipelines have run and all tables have been joined onto the labels frame.
It computes interaction features that span multiple tables.

Configured via ``[data.cross] transformer`` in data.toml — same pattern as
every other processing step.  Set to ``"NoOpStep"`` to disable cross features,
``"CrossTableTransformer"`` to enable.
"""

from __future__ import annotations

import logging

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

_DEFAULT_ID_COLUMN = "SK_ID_CURR"


class CrossTableTransformer(TransformerMixin, BaseEstimator):
    """Cross-table interaction features computed from the merged DataFrame.

    Operates after _join_all() in TableTransformer: receives a single merged
    Polars DataFrame containing prefixed columns from all processed tables and
    returns a dict with a ``"cross"`` key holding a DataFrame of cross features
    (id column + derived columns).

    The id_column is injected by TableTransformer at runtime — no constructor
    argument needed.  Features are computed conditionally on column presence,
    so the transformer is safe with any subset of tables.
    """

    def __init__(self) -> None:
        # id_column is set by TableTransformer before transform() is called.
        self.id_column: str = _DEFAULT_ID_COLUMN

    def fit(self, merged_df: pl.DataFrame, y: object = None) -> CrossTableTransformer:
        """No-op — cross features are stateless aggregations."""
        return self

    def transform(self, merged_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Compute cross-table features from the merged DataFrame.

        Args:
            merged_df: Merged DataFrame with prefixed columns from all tables.

        Returns:
            Dict with ``"cross"`` key containing a DataFrame of
            ``(id_column, cross_feature_1, ...)``.
        """
        if merged_df.is_empty():
            return {
                "cross": merged_df.select(self.id_column).with_columns(
                    pl.lit(None).cast(pl.Float64).alias("cross_dummy")
                )
            }

        col_names = set(merged_df.columns)
        bureau_cols = [c for c in col_names if "bureau_" in c and "bureau_balance" not in c]
        app_cols = [c for c in col_names if c.startswith("application_")]
        bb_cols = [c for c in col_names if c.startswith("bureau_balance_")]
        cc_cols = [c for c in col_names if c.startswith("credit_card_balance_")]
        pos_cols = [c for c in col_names if c.startswith("pos_cash_balance_")]

        logger.debug(
            "cross: bureau=%d app=%d bb=%d cc=%d pos=%d",
            len(bureau_cols),
            len(app_cols),
            len(bb_cols),
            len(cc_cols),
            len(pos_cols),
        )

        result_exprs: list[pl.Expr] = []

        if bureau_cols and app_cols:
            result_exprs.extend(self._bureau_app_cross(merged_df, col_names))
        if bureau_cols and bb_cols:
            result_exprs.extend(self._bureau_bb_cross(merged_df, col_names))
        if cc_cols and bureau_cols:
            result_exprs.extend(self._bureau_cc_cross(merged_df, col_names))
        if pos_cols and bureau_cols:
            result_exprs.extend(self._bureau_pos_cross(merged_df, col_names))
        if cc_cols and pos_cols:
            result_exprs.extend(self._cc_pos_cross(merged_df, col_names))

        if not result_exprs:
            result_exprs.append(pl.lit(None).cast(pl.Float64).alias("cross_dummy"))

        logger.debug("cross: %d features produced", len(result_exprs))

        ids = merged_df.select(self.id_column)
        features = merged_df.select(result_exprs)
        return {"cross": pl.concat([ids, features], how="horizontal")}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bureau_app_cross(self, df: pl.DataFrame, col_names: set[str]) -> list[pl.Expr]:
        """Bureau × Application interactions."""
        exprs: list[pl.Expr] = []

        if "bureau_AMT_CREDIT_SUM_sum" in col_names and "application_AMT_CREDIT" in col_names:
            exprs.append(
                (
                    df["bureau_AMT_CREDIT_SUM_sum"].fill_null(0)
                    / df["application_AMT_CREDIT"].clip(lower_bound=1)
                ).alias("cross_bureau_app_credit_ratio")
            )

        if "bureau_active_ratio" in col_names and "application_ext_source_mean" in col_names:
            exprs.append(
                (
                    df["bureau_active_ratio"] * df["application_ext_source_mean"].fill_null(0.5)
                ).alias("cross_active_stress")
            )

        if "bureau_n_records" in col_names and "application_AMT_INCOME_TOTAL" in col_names:
            exprs.append(
                (
                    df["bureau_n_records"].fill_null(0)
                    / df["application_AMT_INCOME_TOTAL"].clip(lower_bound=1)
                ).alias("cross_bureau_n_records_ratio")
            )

        return exprs

    def _bureau_bb_cross(self, df: pl.DataFrame, col_names: set[str]) -> list[pl.Expr]:
        """Bureau × bureau_balance interactions.

        bureau_* columns are prefixed by TableTransformer from BureauAggregator.
        bureau_balance_bb_* columns are prefixed from BureauBalanceAggregator.
        """
        exprs: list[pl.Expr] = []

        if "bureau_active_ratio" not in col_names:
            return exprs

        bureau_active = df["bureau_active_ratio"]

        # Active credits × ever had a bad loan — joint credit risk signal
        if "bureau_balance_bb_pct_loans_ever_bad" in col_names:
            exprs.append(
                (bureau_active * df["bureau_balance_bb_pct_loans_ever_bad"].fill_null(0)).alias(
                    "cross_bureau_bb_active_ever_bad"
                )
            )

        # Active credits × most recent DPD — current stress signal
        if "bureau_balance_bb_recent_dpd" in col_names:
            exprs.append(
                (bureau_active * df["bureau_balance_bb_recent_dpd"].fill_null(0)).alias(
                    "cross_bureau_bb_active_recent_dpd"
                )
            )

        # Active credits × last 6m max DPD — short-term delinquency stress
        if "bureau_balance_bb_last6m_max_dpd" in col_names:
            exprs.append(
                (bureau_active * df["bureau_balance_bb_last6m_max_dpd"].fill_null(0)).alias(
                    "cross_bureau_bb_active_last6m_dpd"
                )
            )

        # Bureau overdue ratio × bureau_balance delinquency rate
        # — persistent overdue confirmed by payment history
        if (
            "bureau_overdue_ratio" in col_names
            and "bureau_balance_bb_delinquency_rate" in col_names
        ):
            exprs.append(
                (
                    df["bureau_overdue_ratio"].fill_null(0)
                    * df["bureau_balance_bb_delinquency_rate"].fill_null(0)
                ).alias("cross_bureau_overdue_x_delinquency")
            )

        # Total credit exposure × delinquency rate — concentrated risk signal
        if (
            "bureau_AMT_CREDIT_SUM_sum" in col_names
            and "bureau_balance_bb_delinquency_rate" in col_names
        ):
            exprs.append(
                (
                    df["bureau_AMT_CREDIT_SUM_sum"].fill_null(0)
                    * df["bureau_balance_bb_delinquency_rate"].fill_null(0)
                ).alias("cross_bureau_credit_volume_delinquency")
            )

        return exprs

    def _bureau_cc_cross(self, df: pl.DataFrame, col_names: set[str]) -> list[pl.Expr]:
        """Bureau × credit_card_balance interactions."""
        exprs: list[pl.Expr] = []

        if "bureau_active_ratio" not in col_names:
            return exprs

        bureau_active = df["bureau_active_ratio"]

        if "credit_card_balance_cc_utilization_mean" in col_names:
            exprs.append(
                (bureau_active * df["credit_card_balance_cc_utilization_mean"].fill_null(0)).alias(
                    "cross_bureau_credit_card_stress"
                )
            )

        if "credit_card_balance_cc_overlimit_count" in col_names:
            exprs.append(
                (bureau_active * df["credit_card_balance_cc_overlimit_count"].fill_null(0)).alias(
                    "cross_bureau_cc_overlimit"
                )
            )

        return exprs

    def _bureau_pos_cross(self, df: pl.DataFrame, col_names: set[str]) -> list[pl.Expr]:
        """Bureau × pos_cash_balance interactions."""
        exprs: list[pl.Expr] = []

        if "bureau_active_ratio" not in col_names:
            return exprs

        bureau_active = df["bureau_active_ratio"]

        if "pos_cash_balance_pos_dpd_max" in col_names:
            exprs.append(
                (bureau_active * df["pos_cash_balance_pos_dpd_max"].fill_null(0)).alias(
                    "cross_bureau_pos_dpd"
                )
            )

        if "pos_cash_balance_pos_active_count" in col_names:
            exprs.append(
                (bureau_active * df["pos_cash_balance_pos_active_count"].fill_null(0)).alias(
                    "cross_bureau_pos_active"
                )
            )

        return exprs

    def _cc_pos_cross(self, df: pl.DataFrame, col_names: set[str]) -> list[pl.Expr]:
        """Credit card × POS interactions."""
        exprs: list[pl.Expr] = []

        if (
            "credit_card_balance_cc_utilization_mean" in col_names
            and "pos_cash_balance_pos_dpd_max" in col_names
        ):
            exprs.append(
                (
                    df["credit_card_balance_cc_utilization_mean"].fill_null(0)
                    * df["pos_cash_balance_pos_dpd_max"].fill_null(0)
                ).alias("cross_cc_pos_stress")
            )

        return exprs
