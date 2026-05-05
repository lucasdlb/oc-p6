"""Full test suite for data package — all tables, all processing steps."""

from __future__ import annotations

import polars as pl
import pytest

from credit_risk.config import load_config
from credit_risk.data.aggregation.registry import AggregatorRegistry
from credit_risk.data.cleaning.registry import CleaningRegistry
from credit_risk.data.encoding.registry import EncodingRegistry
from credit_risk.data.imputation.registry import ImputationRegistry
from credit_risk.data.transformation.registry import TransformerRegistry


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def loader(cfg):
    from credit_risk.data.loader import PLLazyDataLoader

    return PLLazyDataLoader()


# -------------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------------


def test_loader_bureau(loader):
    df = loader.load("bureau").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_BUREAU" in df.columns
    assert df.height > 0


def test_loader_bureau_balance(loader):
    df = loader.load("bureau_balance").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_BUREAU" in df.columns
    assert "STATUS" in df.columns
    assert df.height > 0


def test_loader_previous_application(loader):
    df = loader.load("previous_application").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_PREV" in df.columns
    assert df.height > 0


def test_loader_pos_cash_balance(loader):
    df = loader.load("pos_cash_balance").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_PREV" in df.columns
    assert "NAME_CONTRACT_STATUS" in df.columns
    assert df.height > 0


def test_loader_installments(loader):
    df = loader.load("installments").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_PREV" in df.columns
    assert "AMT_INSTALMENT" in df.columns
    assert df.height > 0


def test_loader_credit_card_balance(loader):
    df = loader.load("credit_card_balance").collect()
    assert "SK_ID_CURR" in df.columns
    assert "SK_ID_PREV" in df.columns
    assert "AMT_BALANCE" in df.columns
    assert df.height > 0


# -------------------------------------------------------------------------
# Cleaners
# -------------------------------------------------------------------------


def test_cleaner_bureau(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.columns == raw.columns
    assert result.height == raw.height


def test_cleaner_bureau_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.height == raw.height


def test_cleaner_previous_application(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.height == raw.height


def test_cleaner_pos_cash_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.height == raw.height
    assert "NAME_CONTRACT_STATUS" in result.columns


def test_cleaner_installments(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.height == raw.height


def test_cleaner_credit_card_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.height == raw.height


# -------------------------------------------------------------------------
# Imputers
# -------------------------------------------------------------------------


def test_imputer_bureau(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


def test_imputer_bureau_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


def test_imputer_previous_application(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


def test_imputer_pos_cash_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.pos_cash_balance.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


def test_imputer_installments(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


def test_imputer_credit_card_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    result = imputer.fit_transform(cleaned)
    assert result.height == cleaned.height


# -------------------------------------------------------------------------
# Aggregators
# -------------------------------------------------------------------------


def test_aggregator_bureau(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


def test_aggregator_bureau_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau_balance.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


def test_aggregator_previous_application(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.previous_application.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


def test_aggregator_pos_cash_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.pos_cash_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.pos_cash_balance.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


def test_aggregator_installments(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.installments.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


def test_aggregator_credit_card_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.credit_card_balance.aggregator)()
    result = agg.fit_transform(imputed)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0
    assert result.height <= imputed["SK_ID_CURR"].n_unique()


# -------------------------------------------------------------------------
# Aggregator variants
# -------------------------------------------------------------------------


def test_aggregator_installments_variants(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)

    variants = [
        "MinimalInstallmentsAggregator",
        "DefaultInstallmentsAggregator",
        "DetailedInstallmentsAggregator",
    ]
    for name in variants:
        agg = AggregatorRegistry.get(name)()
        result = agg.fit_transform(imputed)
        assert "SK_ID_CURR" in result.columns
        assert result.height > 0


def test_aggregator_bureau_balance_variants(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    variants = [
        "MinimalBureauBalanceAggregator",
        "DefaultBureauBalanceAggregator",
        "DetailedBureauBalanceAggregator",
    ]
    for name in variants:
        agg = AggregatorRegistry.get(name)()
        result = agg.fit_transform(imputed)
        assert "SK_ID_CURR" in result.columns
        assert result.height > 0


def test_aggregator_credit_card_variants(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    variants = [
        "MinimalCreditCardAggregator",
        "DefaultCreditCardAggregator",
        "DetailedCreditCardAggregator",
    ]
    for name in variants:
        agg = AggregatorRegistry.get(name)()
        result = agg.fit_transform(imputed)
        assert "SK_ID_CURR" in result.columns
        assert result.height > 0


# -------------------------------------------------------------------------
# Transformers
# -------------------------------------------------------------------------


def test_transformer_bureau(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.bureau.transformer)()
    result = trans.fit_transform(aggregated)
    assert result.height == aggregated.height


def test_transformer_previous_application(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.previous_application.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.previous_application.transformer)()
    result = trans.fit_transform(aggregated)
    assert result.height == aggregated.height


def test_transformer_installments(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.installments.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.installments.transformer)()
    result = trans.fit_transform(aggregated)
    assert result.height == aggregated.height


def test_transformer_credit_card_balance(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.credit_card_balance.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.credit_card_balance.transformer)()
    result = trans.fit_transform(aggregated)
    assert result.height == aggregated.height


# -------------------------------------------------------------------------
# Transformers — column presence
# -------------------------------------------------------------------------


def test_transformer_bureau_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.bureau.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "bureau_avg_to_max_credit_age",
        "bureau_credit_to_annuity_ratio",
        "bureau_overdue_ratio",
        "bureau_overdue_severity",
        "bureau_avg_prolong_per_credit",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


def test_transformer_previous_application_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.previous_application.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.previous_application.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "prev_credit_fulfilment_rate",
        "prev_annuity_to_credit_ratio",
        "prev_down_payment_rate",
        "prev_refused_to_approved_amt_ratio",
        "prev_applications_per_year",
        "prev_refusal_rate_1y",
        "prev_credit_recency_ratio",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


def test_transformer_installments_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.installments.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.installments.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "ins_payment_to_instalment_ratio",
        "ins_payment_diff",
        "ins_instalment_version_complexity",
        "ins_long_payment_history",
        "ins_late_underpay_interaction",
        "ins_payment_behavior_score",
        "ins_recent_vs_historical_late",
        "ins_avg_payment_per_record",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


def test_transformer_credit_card_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.credit_card_balance.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.credit_card_balance.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "cc_balance_to_limit_ratio",
        "cc_utilization_ratio",
        "cc_total_to_current_payment_ratio",
        "cc_atm_to_total_drawings_ratio",
        "cc_dpd_to_def_ratio",
        "cc_dpd_intensity",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


def test_transformer_pos_cash_balance_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.pos_cash_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.pos_cash_balance.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.pos_cash_balance.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "pos_dpd_to_def_ratio",
        "pos_future_to_total_instalment_ratio",
        "pos_history_span",
        "pos_dpd_intensity",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


def test_transformer_bureau_balance_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau_balance.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.bureau_balance.transformer)()
    result = trans.fit_transform(aggregated)
    expected = {
        "bb_dpd_severity",
        "bb_delinquency_intensity",
        "bb_avg_duration_per_credit",
        "bb_recent_activity_intensity",
        "bb_bad_loan_concentration",
        "bb_improvement_score",
    }
    assert result.height == aggregated.height
    missing = expected - set(result.columns)
    assert not missing, f"Missing transformer columns: {missing}"


# -------------------------------------------------------------------------
# Cleaners — domain logic
# -------------------------------------------------------------------------


def test_cleaner_bureau_sentinels(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    result = cleaner.fit_transform(raw)
    sentinel_cols = ["DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]
    for col in sentinel_cols:
        if col in raw.columns:
            assert result.filter(pl.col(col) == 365243).height == 0, f"Sentinel not removed: {col}"
    amt_cols = ["AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_ANNUITY"]
    for col in amt_cols:
        if col in raw.columns:
            assert result.filter(pl.col(col) < 0).height == 0, f"Negative AMT not removed: {col}"


def test_cleaner_previous_application_sentinels(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    result = cleaner.fit_transform(raw)
    sentinel_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION"]
    for col in sentinel_cols:
        if col in raw.columns:
            assert result.filter(pl.col(col) == 365243).height == 0, f"Sentinel not removed: {col}"
    assert result.filter(pl.col("NAME_CONTRACT_TYPE").is_in(["XNA", "XAP"])).height == 0


def test_cleaner_pos_cash_balance_xna(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    result = cleaner.fit_transform(raw)
    assert result.filter(pl.col("NAME_CONTRACT_STATUS") == "XNA").height == 0


def test_cleaner_credit_card_balance_fillnull(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    result = cleaner.fit_transform(raw)
    null_drawing = result.filter(pl.col("AMT_DRAWINGS_ATM_CURRENT").is_null()).height
    assert null_drawing == 0, "Drawing columns should have no nulls after cleaning"


# -------------------------------------------------------------------------
# Imputers — null-free verification
# -------------------------------------------------------------------------


def test_imputer_bureau_nullfree(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    result = imputer.fit_transform(cleaned)
    num_cols = [c for c in result.columns if result[c].dtype in (pl.Float64, pl.Int64, pl.Int32)]
    null_counts = {c: result[c].null_count() for c in num_cols}
    problematic = {c: n for c, n in null_counts.items() if n > 0}
    assert not problematic, f"Nulls remaining after imputation: {problematic}"


def test_imputer_previous_application_nullfree(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    result = imputer.fit_transform(cleaned)
    num_cols = [c for c in result.columns if result[c].dtype in (pl.Float64, pl.Int64, pl.Int32)]
    null_counts = {c: result[c].null_count() for c in num_cols}
    problematic = {c: n for c, n in null_counts.items() if n > 0}
    assert not problematic, f"Nulls remaining after imputation: {problematic}"


def test_imputer_pos_cash_balance_nullfree(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.pos_cash_balance.imputer)()
    result = imputer.fit_transform(cleaned)
    num_cols = [c for c in result.columns if result[c].dtype in (pl.Float64, pl.Int64, pl.Int32)]
    null_counts = {c: result[c].null_count() for c in num_cols}
    problematic = {c: n for c, n in null_counts.items() if n > 0}
    assert not problematic, f"Nulls remaining after imputation: {problematic}"


def test_imputer_installments_nullfree(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    result = imputer.fit_transform(cleaned)
    num_cols = [c for c in result.columns if result[c].dtype in (pl.Float64, pl.Int64, pl.Int32)]
    null_counts = {c: result[c].null_count() for c in num_cols}
    problematic = {c: n for c, n in null_counts.items() if n > 0}
    assert not problematic, f"Nulls remaining after imputation: {problematic}"


def test_imputer_credit_card_balance_nullfree(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    result = imputer.fit_transform(cleaned)
    num_cols = [c for c in result.columns if result[c].dtype in (pl.Float64, pl.Int64, pl.Int32)]
    null_counts = {c: result[c].null_count() for c in num_cols}
    problematic = {c: n for c, n in null_counts.items() if n > 0}
    assert not problematic, f"Nulls remaining after imputation: {problematic}"


# -------------------------------------------------------------------------
# Aggregators — domain column presence
# -------------------------------------------------------------------------


def test_aggregator_bureau_domain_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau.aggregator)()
    result = agg.fit_transform(imputed)
    expected = {
        "bureau_n_records",
        "bureau_active_count",
        "bureau_closed_count",
        "bureau_overdue_count",
        "bureau_earliest_credit_days",
        "bureau_credit_span_days",
        "bureau_CREDIT_ACTIVE_n_unique",
        "bureau_CREDIT_TYPE_n_unique",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing aggregator columns: {missing}"


def test_aggregator_previous_application_domain_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("previous_application")
    cleaner = CleaningRegistry.get(cfg.data.previous_application.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.previous_application.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.previous_application.aggregator)()
    result = agg.fit_transform(imputed)
    expected = {
        "prev_n_records",
        "prev_approved_count",
        "prev_refused_count",
        "prev_canceled_count",
        "prev_approval_rate",
        "prev_refusal_rate",
        "prev_client_new_count",
        "prev_yield_group_mean",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing aggregator columns: {missing}"


def test_aggregator_pos_cash_balance_domain_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("pos_cash_balance")
    cleaner = CleaningRegistry.get(cfg.data.pos_cash_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.pos_cash_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.pos_cash_balance.aggregator)()
    result = agg.fit_transform(imputed)
    expected = {
        "pos_n_records",
        "pos_dpd_max",
        "pos_dpd_months_count",
        "pos_completed_count",
        "pos_active_count",
        "pos_demand_count",
        "pos_demand_rate",
        "pos_completion_rate",
        "pos_dpd_mean_recent",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing aggregator columns: {missing}"


def test_aggregator_installments_domain_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.installments.aggregator)()
    result = agg.fit_transform(imputed)
    expected = {
        "ins_n_records",
        "ins_ever_late",
        "ins_late_payment_rate",
        "ins_underpayment_rate",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing aggregator columns: {missing}"


def test_aggregator_credit_card_balance_domain_columns(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.credit_card_balance.aggregator)()
    result = agg.fit_transform(imputed)
    expected = {
        "cc_n_records",
        "cc_completed_count",
        "cc_active_count",
        "cc_completion_rate",
        "cc_ever_overdue",
        "cc_utilization_mean",
        "cc_utilization_recent",
        "cc_dpd_max",
    }
    missing = expected - set(result.columns)
    assert not missing, f"Missing aggregator columns: {missing}"


# -------------------------------------------------------------------------
# Aggregator variants — column count monotonicity
# -------------------------------------------------------------------------


def test_aggregator_installments_variant_column_count(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("installments")
    cleaner = CleaningRegistry.get(cfg.data.installments.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.installments.imputer)()
    imputed = imputer.fit_transform(cleaned)

    minimal = AggregatorRegistry.get("MinimalInstallmentsAggregator")().fit_transform(imputed)
    default = AggregatorRegistry.get("DefaultInstallmentsAggregator")().fit_transform(imputed)
    detailed = AggregatorRegistry.get("DetailedInstallmentsAggregator")().fit_transform(imputed)

    assert len(default.columns) > len(minimal.columns)
    assert len(detailed.columns) > len(default.columns)


def test_aggregator_credit_card_variant_column_count(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    minimal = AggregatorRegistry.get("MinimalCreditCardAggregator")().fit_transform(imputed)
    default = AggregatorRegistry.get("DefaultCreditCardAggregator")().fit_transform(imputed)

    assert len(default.columns) > len(minimal.columns)


def test_aggregator_bureau_balance_variant_column_count(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    minimal = AggregatorRegistry.get("MinimalBureauBalanceAggregator")().fit_transform(imputed)
    default = AggregatorRegistry.get("DefaultBureauBalanceAggregator")().fit_transform(imputed)
    detailed = AggregatorRegistry.get("DetailedBureauBalanceAggregator")().fit_transform(imputed)

    assert len(default.columns) > len(minimal.columns)
    assert len(detailed.columns) > len(default.columns)


def test_bureau_balance_new_features_present(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau_balance")
    cleaner = CleaningRegistry.get(cfg.data.bureau_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    default = AggregatorRegistry.get("DefaultBureauBalanceAggregator")().fit_transform(imputed)
    detailed = AggregatorRegistry.get("DetailedBureauBalanceAggregator")().fit_transform(imputed)

    assert "bb_closed_rate" in default.columns, "bb_closed_rate should be in Default"
    assert "bb_mean_dpd_recent" in default.columns, "bb_mean_dpd_recent should be in Default"
    assert "bb_mean_dpd_recent" in detailed.columns, "bb_mean_dpd_recent should be in Detailed"
    assert "bb_slope_reversal" in detailed.columns, "bb_slope_reversal should be in Detailed"


def test_credit_card_utilization_recent_present(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("credit_card_balance")
    cleaner = CleaningRegistry.get(cfg.data.credit_card_balance.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.credit_card_balance.imputer)()
    imputed = imputer.fit_transform(cleaned)

    default = AggregatorRegistry.get("DefaultCreditCardAggregator")().fit_transform(imputed)
    assert "cc_utilization_recent" in default.columns, "cc_utilization_recent should be in Default"


# -------------------------------------------------------------------------
# Full pipeline — remaining tables
# -------------------------------------------------------------------------


def test_pipeline_bureau_balance(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("bureau_balance")
    pipe = ProcessingPipeline(cfg.data.bureau_balance).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


def test_pipeline_pos_cash_balance(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("pos_cash_balance")
    pipe = ProcessingPipeline(cfg.data.pos_cash_balance).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


# -------------------------------------------------------------------------
# Schema enforcement
# -------------------------------------------------------------------------


def test_encoder_onehot(cfg):
    from credit_risk.data.loader import PLDataLoader

    raw = PLDataLoader().load("bureau")
    cleaner = CleaningRegistry.get(cfg.data.bureau.cleaner)()
    cleaned = cleaner.fit_transform(raw)
    imputer = ImputationRegistry.get(cfg.data.bureau.imputer)()
    imputed = imputer.fit_transform(cleaned)
    agg = AggregatorRegistry.get(cfg.data.bureau.aggregator)()
    aggregated = agg.fit_transform(imputed)
    trans = TransformerRegistry.get(cfg.data.bureau.transformer)()
    transformed = trans.fit_transform(aggregated)
    enc = EncodingRegistry.get(cfg.data.bureau.encoder)()
    result = enc.fit_transform(transformed)
    assert result.height == transformed.height


# -------------------------------------------------------------------------
# Full pipeline
# -------------------------------------------------------------------------


def test_pipeline_credit_card_balance(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("credit_card_balance")
    pipe = ProcessingPipeline(cfg.data.credit_card_balance).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


def test_pipeline_installments(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("installments")
    pipe = ProcessingPipeline(cfg.data.installments).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


def test_pipeline_previous_application(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("previous_application")
    pipe = ProcessingPipeline(cfg.data.previous_application).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


def test_pipeline_bureau(cfg):
    from credit_risk.data.loader import PLDataLoader
    from credit_risk.pipeline.processing_pipeline import ProcessingPipeline

    raw = PLDataLoader().load("bureau")
    pipe = ProcessingPipeline(cfg.data.bureau).build()
    result = pipe.fit_transform(raw)
    assert "SK_ID_CURR" in result.columns
    assert result.height > 0


# -------------------------------------------------------------------------
# Schema enforcement
# -------------------------------------------------------------------------


def test_schema_enforcer():
    import polars as pl

    from credit_risk.data.base import SchemaEnforcer

    enforcer = SchemaEnforcer()
    df = pl.DataFrame({"SK_ID_CURR": [1, 2], "a": [3, 4]})
    result = enforcer.fit_transform(df)
    assert "SK_ID_CURR" in result.columns
    assert result["SK_ID_CURR"].dtype == pl.Int64
