#!/usr/bin/env python
"""Data pipeline diagnostic script - full column logging."""

from __future__ import annotations

from credit_risk.config import load_config
from credit_risk.data.aggregation.registry import AggregatorRegistry
from credit_risk.data.cleaning.registry import CleaningRegistry
from credit_risk.data.imputation.registry import ImputationRegistry
from credit_risk.data.loader import PLDataLoader


def run_pipeline_for_table(table_name: str, config) -> dict | None:
    """Run full pipeline for a table and return diagnostics."""
    loader = PLDataLoader()

    try:
        raw = loader.load(table_name)
    except Exception as e:
        print(f"  [LOAD FAILED for {table_name}: {e}")
        return None

    results = {
        "table": table_name,
        "raw": {"rows": raw.height, "cols": len(raw.columns), "col_list": list(raw.columns)},
    }

    # Get processing config
    table_cfg = getattr(config.data, table_name, None)
    if table_cfg is None:
        print(f"  No config for {table_name}, skipping")
        return None

    cleaner_cfg = getattr(table_cfg, "cleaner", "default")
    imputer_cfg = getattr(table_cfg, "imputer", "default")
    aggregator_cfg = getattr(table_cfg, "aggregator", "default")

    print(
        f"  {table_name}: cleaner={cleaner_cfg}, imputer={imputer_cfg}, aggregator={aggregator_cfg}"
    )

    # Step 1: Cleaning
    try:
        cleaner = CleaningRegistry.get(cleaner_cfg)()
        cleaned = cleaner.fit_transform(raw)
        results["cleaned"] = {
            "rows": cleaned.height,
            "cols": len(cleaned.columns),
            "col_list": list(cleaned.columns),
            "added": [c for c in cleaned.columns if c not in raw.columns],
            "removed": [c for c in raw.columns if c not in cleaned.columns],
        }
    except Exception as e:
        print(f"  [CLEANING FAILED for {table_name}: {e}")
        results["cleaned"] = {"rows": 0, "cols": 0, "added": [], "removed": [], "col_list": []}

    # Step 2: Imputation
    try:
        imputer = ImputationRegistry.get(imputer_cfg)()
        imputed = imputer.fit_transform(cleaned)
        results["imputed"] = {
            "rows": imputed.height,
            "cols": len(imputed.columns),
            "col_list": list(imputed.columns),
            "added": [c for c in imputed.columns if c not in cleaned.columns],
            "removed": [c for c in cleaned.columns if c not in imputed.columns],
        }
    except Exception as e:
        print(f"  [IMPUTATION FAILED for {table_name}: {e}")
        results["imputed"] = {"rows": 0, "cols": 0, "added": [], "removed": [], "col_list": []}

    # Step 3: Aggregation
    try:
        aggregator = AggregatorRegistry.get(aggregator_cfg)()
        aggregated = aggregator.fit_transform(imputed)
        results["aggregated"] = {
            "rows": aggregated.height,
            "cols": len(aggregated.columns),
            "col_list": list(aggregated.columns),
            "added": [c for c in aggregated.columns if c not in imputed.columns],
            "removed": [c for c in imputed.columns if c not in aggregated.columns],
        }
    except Exception as e:
        print(f"  [AGGREGATION FAILED for {table_name}: {e}")
        results["aggregated"] = {"rows": 0, "cols": 0, "added": [], "removed": [], "col_list": []}

    return results


def display_full_columns(results: dict):
    """Display all columns for each step."""
    print(f"\n{'=' * 60}")
    print(f"Table: {results['table']}")
    print(f"{'=' * 60}")

    # Show raw columns
    step = "raw"
    data = results.get(step, {})
    print(f"\n--- {step.upper()} ({data.get('rows', 0)} rows, {data.get('cols', 0)} cols) ---")
    print(data.get("col_list", []))

    # Show cleaned columns
    step = "cleaned"
    data = results.get(step, {})
    if data.get("col_list"):
        print(f"\n--- {step.upper()} ({data.get('rows', 0)} rows, {data.get('cols', 0)} cols) ---")
        print(f"Added: {data.get('added', [])}")
        print(f"Removed: {data.get('removed', [])}")
        print(data.get("col_list", []))

    # Show imputed columns
    step = "imputed"
    data = results.get(step, {})
    if data.get("col_list"):
        print(f"\n--- {step.upper()} ({data.get('rows', 0)} rows, {data.get('cols', 0)} cols) ---")
        print(f"Added: {data.get('added', [])}")
        print(f"Removed: {data.get('removed', [])}")
        print(data.get("col_list", []))

    # Show aggregated columns
    step = "aggregated"
    data = results.get(step, {})
    if data.get("col_list"):
        print(f"\n--- {step.upper()} ({data.get('rows', 0)} rows, {data.get('cols', 0)} cols) ---")
        print(f"Added: {data.get('added', [])}")
        print(f"Removed: {data.get('removed', [])}")
        print(data.get("col_list", []))


def main():
    config = load_config()

    # Process each table
    all_table_names = [
        "application",
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "pos_cash_balance",
        "installments",
        "previous_application",
    ]

    print("=" * 60)
    print("DATA PIPELINE DIAGNOSTICS - FULL COLUMNS")
    print("=" * 60)

    for table_name in all_table_names:
        try:
            print(f"\n>>> Processing {table_name}...")
            result = run_pipeline_for_table(table_name, config)
            if result:
                display_full_columns(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
