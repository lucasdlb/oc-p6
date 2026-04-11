"""Audit all data tables to find schema issues - bypasses KNOWN_SCHEMA_OVERRIDES."""

import polars as pl

from credit_risk.config import cfg
from credit_risk.data.loader import TABLES_CSV_NAMES


def audit_table_raw(name: str) -> dict[str, type]:
    """Load a table WITHOUT schema overrides and find numeric columns read as string."""
    from credit_risk.config.config import PROJECT_ROOT

    data_path = PROJECT_ROOT / cfg.data.data_dir
    path = data_path / TABLES_CSV_NAMES[name]

    # Load WITHOUT any schema overrides
    df = pl.read_csv(path)

    issues: dict[str, type] = {}

    for col in df.columns:
        if df.schema[col] == pl.Utf8:
            try:
                casted = df[col].cast(pl.Float64, strict=False)
                null_before = df[col].is_null().sum()
                null_after = casted.is_null().sum()
                if null_before == null_after:
                    issues[col] = pl.Float64
            except Exception:
                pass

    return issues


def main():
    tables = list(TABLES_CSV_NAMES.keys())

    print("=" * 60)
    print("RAW SCHEMA AUDIT (no overrides)")
    print("=" * 60)

    all_overrides: dict[str, dict[str, type]] = {}

    for table in tables:
        print(f"\n{table}:")
        try:
            issues = audit_table_raw(table)
            if issues:
                print(f"  Issues: {list(issues.keys())}")
                all_overrides[table] = issues
            else:
                print("  OK - no schema issues")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("FINAL KNOWN_SCHEMA_OVERRIDES")
    print("=" * 60)
    print()
    if all_overrides:
        print("KNOWN_SCHEMA_OVERRIDES: dict[str, dict[str, type[pl.DataType]]] = {")
        for table, issues in all_overrides.items():
            print(f'    "{table}": {{')
            for col, dtype in issues.items():
                print(f'        "{col}": pl.{dtype},')
            print("    },")
        print("}")
    else:
        print("# No schema overrides needed - all columns loaded correctly")


if __name__ == "__main__":
    main()
