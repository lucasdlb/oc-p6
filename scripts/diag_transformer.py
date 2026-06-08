#!/usr/bin/env python
"""Diagnostic script using TableTransformer methods to investigate cross transformer issue."""

from __future__ import annotations

import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLDataLoader
from credit_risk_processing.data.transformation import TransformerRegistry
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.pipeline.table_transformer import TableTransformer


def main():
    cfg = load_config()
    loader = PLDataLoader()

    print("=" * 60)
    print("TABLE TRANSFORMER DIAGNOSTIC - Using TableTransformer")
    print("=" * 60)

    # Load tables
    print("\n=== STEP 1: Load tables ===")
    app = loader.load("application")
    bureau = loader.load("bureau")
    print(f"application: {app.height} rows")
    print(f"bureau: {bureau.height} rows")

    # Create labels
    labels = app.select("SK_ID_CURR", "TARGET")
    labels = labels.with_columns(
        pl.when(pl.col("TARGET") == 1).then(1).otherwise(0).alias("TARGET")
    )

    # Get IDs
    all_ids = app["SK_ID_CURR"].to_numpy()
    train_ids = set(all_ids[:500])
    val_ids = set(all_ids[500:1000])

    print(f"train_ids: {len(train_ids)}, val_ids: {len(val_ids)}")

    # Create pipeline factories (same as experiment.py)
    print("\n=== STEP 2: Create pipeline_factories ===")
    tables = ["application", "bureau"]
    pipeline_factories = {
        t: lambda tbl=t: ProcessingPipeline(getattr(cfg.data, tbl)).build() for t in tables
    }

    cross_cls = TransformerRegistry.get("CrossTableTransformer")
    cross_transformer = cross_cls(id_column="SK_ID_CURR")

    tt = TableTransformer(
        pipeline_factories=pipeline_factories,
        id_column="SK_ID_CURR",
        target_column="TARGET",
        cross_transformer=cross_transformer,
    )
    print("TableTransformer created")
    print(f"  cross_transformer: {tt.cross_transformer}")

    # Process tables (similar to _process_tables)
    print("\n=== STEP 3: Process tables ===")
    tables_raw = {"application": app, "bureau": bureau}

    train_dfs, val_dfs = {}, {}

    for name, raw_df in tables_raw.items():
        print(f"\n--- Processing {name} ---")

        df_train_raw = raw_df.filter(pl.col(tt.id_column).is_in(train_ids))
        df_val_raw = raw_df.filter(pl.col(tt.id_column).is_in(val_ids))

        y_train_raw = labels.filter(
            pl.col(tt.id_column).is_in(df_train_raw.select(tt.id_column).to_series())
        )
        y_train = y_train_raw.select(tt.target_column).to_numpy().ravel()

        if tt.target_column in df_train_raw.columns:
            df_train_raw = df_train_raw.drop(tt.target_column)
            df_val_raw = df_val_raw.drop(tt.target_column)

        # Get pipeline
        pipe = pipeline_factories[name]()
        pipe.fit(df_train_raw, y=y_train)

        train_out = pipe.transform(df_train_raw)
        val_out = pipe.transform(df_val_raw)

        # Prefix columns
        train_prefixed = tt._prefix_columns(train_out, name)
        val_prefixed = tt._prefix_columns(val_out, name)

        train_dfs[name] = train_prefixed
        val_dfs[name] = val_prefixed

        print(f"  train: {train_prefixed.height} rows × {len(train_prefixed.columns)} cols")
        print(f"  val: {val_prefixed.height} rows × {len(val_prefixed.columns)} cols")

        # Check entropy columns
        entropy_cols = [c for c in train_prefixed.columns if "_entropy" in c.lower()]
        print(f"  entropy cols: {entropy_cols[:3]}...")

    # Join (similar to _merge_and_convert)
    print("\n=== STEP 4: Join tables ===")
    labels_train = labels.filter(pl.col(tt.id_column).is_in(train_ids))
    labels_val = labels.filter(pl.col(tt.id_column).is_in(val_ids))

    merged_train = tt._join_all(labels_train, train_dfs)
    merged_val = tt._join_all(labels_val, val_dfs)

    print(f"merged_train: {merged_train.height} rows × {len(merged_train.columns)} cols")
    print(f"merged_val: {merged_val.height} rows × {len(merged_val.columns)} cols")

    # Check merged_train schema
    print(f"\nmerged_train sample cols: {merged_train.columns[:10]}")
    print(f"merged_train id dtype: {merged_train[tt.id_column].dtype}")

    # Check bureau columns
    bureau_cols = [c for c in merged_train.columns if c.startswith("bureau_")]
    print(f"\nbureau cols ({len(bureau_cols)}): {bureau_cols[:5]}...")

    entropy_cols = [c for c in merged_train.columns if "_entropy" in c.lower()]
    print(f"entropy cols: {entropy_cols}")

    # Apply cross transformer
    print("\n=== STEP 5: Apply cross transformer ===")
    print(f"cross_transformer: {tt.cross_transformer}")
    print("calling cross_transformer.transform(merged_train)...")

    cross_result = tt.cross_transformer.transform(merged_train)
    print(f"cross_result keys: {cross_result.keys()}")

    if "cross" in cross_result:
        cross_df = cross_result["cross"]
        print(f"cross_df: {cross_df.height} rows × {len(cross_df.columns)} cols")
        print(f"cross_df cols: {cross_df.columns}")
        print(f"cross_df id column: {tt.id_column in cross_df.columns}")
        col = tt.id_column
        dtype = cross_df[col].dtype if col in cross_df.columns else "N/A"
        print(f"cross_df {col} dtype: {dtype}")
    else:
        print("WARNING: no 'cross' key!")

    # Try the join that fails in pipeline
    print("\n=== STEP 6: Try join ===")
    if "cross" in cross_result:
        cross_df = cross_result["cross"]
        cross_cols = [c for c in cross_df.columns if c != tt.id_column]
        print(f"cross cols (excluding id): {cross_cols}")

        if cross_cols:
            print("Attempting join...")
            try:
                cross_to_join = cross_df.select([tt.id_column] + cross_cols)
                print(f"cross_to_join cols: {cross_to_join.columns}")

                # Check both dtypes
                print(f"left id dtype: {merged_train[tt.id_column].dtype}")
                print(f"right id dtype: {cross_to_join[tt.id_column].dtype}")

                # Try join
                result = merged_train.join(
                    cross_to_join,
                    left_on=tt.id_column,
                    right_on=tt.id_column,
                    how="left",
                )
                print(f"SUCCESS: result {result.height} rows × {len(result.columns)} cols")
            except Exception as e:
                print(f"JOIN FAILED: {type(e).__name__}: {e}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
