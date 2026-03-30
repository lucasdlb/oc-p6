"""Exploratory Data Analysis for Credit Risk Dataset."""

from __future__ import annotations

import polars as pl

from credit_risk.data import loader


def analyze_application_data() -> None:
    """Analyze the main application data."""
    print("=" * 60)
    print("APPLICATION DATA ANALYSIS")
    print("=" * 60)

    df = loader.load_application(train=True)
    print(f"\nShape: {df.shape}")

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Target distribution ---")
    target_dist = df.group_by("TARGET").len()
    print(target_dist)
    default_rate = df.select(pl.col("TARGET").mean()).item()
    print(f"\nDefault rate: {default_rate:.2%}")

    print("\n--- Numerical columns summary ---")
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]
    print(df.select(numeric_cols).describe())

    print("\n--- Missing values ---")
    missing = df.select([pl.col(c).is_null().sum().alias(c) for c in df.columns]).transpose()
    missing.columns = ["count"]
    missing = missing.filter(pl.col("count") > 0)
    print(missing)


def analyze_previous_application() -> None:
    """Analyze previous application data."""
    print("\n" + "=" * 60)
    print("PREVIOUS APPLICATION DATA ANALYSIS")
    print("=" * 60)

    df = loader.load_previous_application()
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns}")

    print("\n--- Contract status distribution ---")
    status_dist = df.group_by("NAME_CONTRACT_STATUS").len()
    print(status_dist)


def analyze_bureau() -> None:
    """Analyze bureau data."""
    print("\n" + "=" * 60)
    print("BUREAU DATA ANALYSIS")
    print("=" * 60)

    df = loader.load_bureau()
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns}")

    print("\n--- Credit active distribution ---")
    active_dist = df.group_by("CREDIT_ACTIVE").len()
    print(active_dist)


def analyze_related_tables() -> None:
    """Analyze other related tables."""
    tables = [
        ("installments_payments", loader.load_installments_payments),
        ("bureau_balance", loader.load_bureau_balance),
        ("pos_cash_balance", loader.load_pos_cash_balance),
        ("credit_card_balance", loader.load_credit_card_balance),
    ]

    for name, load_fn in tables:
        df = load_fn()
        print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} columns")


def main() -> None:
    """Run all exploratory analyses."""
    print("Starting Exploratory Data Analysis...")
    print(f"Data path: {loader.get_data_path()}\n")

    analyze_application_data()
    analyze_previous_application()
    analyze_bureau()
    analyze_related_tables()

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
