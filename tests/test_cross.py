"""Tests for cross-table feature engineering."""

from __future__ import annotations

import polars as pl
import pytest

from credit_risk.data.transformation.cross import CrossTableTransformer


class TestCrossTableTransformer:
    def test_single_table_no_cross_features(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "application_AMT_CREDIT": [100000, 200000, 150000],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "SK_ID_CURR" in cross_df.columns
        assert cross_df.height == 3

    def test_two_tables_with_cross_features(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "application_AMT_CREDIT": [100000, 200000, 150000],
                "application_ext_source_mean": [0.5, 0.3, 0.7],
                "bureau_AMT_CREDIT_SUM_sum": [50000, 0, 100000],
                "bureau_active_ratio": [0.5, 0.0, 1.0],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_bureau_app_credit_ratio" in cross_df.columns
        assert "cross_active_stress" in cross_df.columns

    def test_bureau_cc_cross(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "bureau_active_ratio": [0.5, 1.0],
                "credit_card_balance_cc_utilization_mean": [0.3, 0.8],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_bureau_credit_card_stress" in cross_df.columns

    def test_bureau_pos_cross(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "bureau_active_ratio": [0.5, 1.0],
                "pos_cash_balance_pos_dpd_max": [0, 2],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_bureau_pos_dpd" in cross_df.columns

    def test_bureau_bb_cross(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "bureau_active_ratio": [0.5, 1.0],
                "bureau_balance_bb_pct_loans_ever_bad": [0.2, 0.8],
                "bureau_balance_bb_recent_dpd": [0, 3],
                "bureau_balance_bb_last6m_max_dpd": [0, 2],
                "bureau_overdue_ratio": [0.1, 0.5],
                "bureau_balance_bb_delinquency_rate": [0.1, 0.4],
                "bureau_AMT_CREDIT_SUM_sum": [50000.0, 200000.0],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_bureau_bb_active_ever_bad" in cross_df.columns
        assert "cross_bureau_bb_active_recent_dpd" in cross_df.columns
        assert "cross_bureau_bb_active_last6m_dpd" in cross_df.columns
        assert "cross_bureau_overdue_x_delinquency" in cross_df.columns
        assert "cross_bureau_credit_volume_delinquency" in cross_df.columns
        assert cross_df.height == 2

    def test_cc_pos_cross(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "credit_card_balance_cc_utilization_mean": [0.3, 0.8],
                "pos_cash_balance_pos_dpd_max": [0, 2],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_cc_pos_stress" in cross_df.columns

    def test_empty_dataframe_returns_dummy(self):
        merged = pl.DataFrame({"SK_ID_CURR": []}).cast({"SK_ID_CURR": pl.Int64})

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_dummy" in cross_df.columns

    def test_no_matching_columns_returns_dummy(self):
        merged = pl.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "some_unrelated_col": [1.0, 2.0],
            }
        )

        transformer = CrossTableTransformer()
        result = transformer.transform(merged)
        cross_df = result["cross"]

        assert "cross_dummy" in cross_df.columns

    def test_transformer_mixin_interface(self):
        transformer = CrossTableTransformer()
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")

        merged = pl.DataFrame({"SK_ID_CURR": [1, 2]})
        result = transformer.transform(merged)
        assert isinstance(result, dict)
        assert "cross" in result

    def test_registered_in_transformer_registry(self):
        from credit_risk.data.transformation import TransformerRegistry

        assert "CrossTableTransformer" in TransformerRegistry.available()

    def test_noop_resolves_to_noop(self):
        from credit_risk.data.base import NoOpStep
        from credit_risk.data.transformation import TransformerRegistry

        cls = TransformerRegistry.get("NoOpStep")
        assert cls is NoOpStep

    def test_unknown_key_raises(self):
        from credit_risk.data.transformation import TransformerRegistry

        with pytest.raises(KeyError, match="not found"):
            TransformerRegistry.get("NonExistentTransformer")
