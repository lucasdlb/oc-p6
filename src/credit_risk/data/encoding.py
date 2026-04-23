"""Categorical encoder for string columns."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

"""One-hot encoder with null and rare-category handling."""


from dataclasses import dataclass, field

import polars as pl


_MISSING = "__missing__"
_RARE = "__rare__"

@dataclass
class CategoricalEncoder:
    """Stateful fit/transform encoder for string columns.

    Fits a stable vocabulary on training data, then applies it
    consistently to any subsequent data (validation, test, inference).

    Encoding strategy per column:
    - Binary columns (2 unique non-null values): 0/1, nulls → -1
    - Low cardinality (≤ threshold): label encoding with stable sort, nulls → -1
    - High cardinality (> threshold): frequency encoding on train distribution

    Example
    -------
    >>> encoder = CategoricalEncoder()
    >>> train = encoder.fit_transform(train_df)
    >>> test  = encoder.transform(test_df)
    """

    high_cardinality_threshold: int = 10

    _vocabularies: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    _freq_maps: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _strategies: dict[str, str] = field(default_factory=dict, init=False)

    def fit(self, df: pl.DataFrame) -> "CategoricalEncoder":
        """Learn vocabularies from df. Call only on training data."""
        string_cols = [c for c, t in df.schema.items() if t == pl.String]

        for col in string_cols:
            series = df[col].drop_nulls()
            unique = sorted(series.unique().to_list())
            n_unique = len(unique)

            if n_unique <= 2:
                self._vocabularies[col] = {v: i for i, v in enumerate(unique)}
                self._strategies[col] = "binary"

            elif n_unique <= self.high_cardinality_threshold:
                self._vocabularies[col] = {v: i for i, v in enumerate(unique)}
                self._strategies[col] = "label"

            else:
                counts = df[col].value_counts(sort=True)
                total = len(df)
                self._freq_maps[col] = {row[col]: row["count"] / total for row in counts.to_dicts()}
                self._strategies[col] = "frequency"

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted encoding. Unknown categories → -1 (label) or 0.0 (freq)."""
        if not self._strategies:
            raise RuntimeError("Call fit() before transform().")

        exprs = []
        for col, strategy in self._strategies.items():
            if col not in df.columns:
                continue

            if strategy in ("binary", "label"):
                vocab = self._vocabularies[col]
                values = [vocab.get(v, -1) if v is not None else -1 for v in df[col].to_list()]
                mapping_series = pl.Series(name=col, values=values, dtype=pl.Int16)
                exprs.append(mapping_series)

            elif strategy == "frequency":
                freq_map = self._freq_maps[col]
                values = [
                    freq_map.get(v, 0.0) if v is not None else float("nan")
                    for v in df[col].to_list()
                ]
                freq_series = pl.Series(name=col, values=values, dtype=pl.Float32)
                exprs.append(freq_series)

        return df.with_columns(exprs)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)

    def unknown_categories(self, df: pl.DataFrame) -> dict[str, list[str]]:
        """Audit: return categories in df not seen during fit."""
        report = {}
        for col, strategy in self._strategies.items():
            if col not in df.columns or strategy == "frequency":
                continue
            known = set(self._vocabularies[col])
            unseen = set(df[col].drop_nulls().unique().to_list()) - known
            if unseen:
                report[col] = sorted(unseen)
        return report




@dataclass
class PolarsOneHotEncoder:
    """One-hot encoder using Polars native operations.

    Encoding rules (all applied at fit time, consistently at transform time):

    - Nulls and NaNs are treated as a dedicated category ``__missing__``.
    - Categories whose training frequency is strictly below
      ``rare_threshold`` are collapsed into ``__rare__``.
    - Columns with more than ``max_categories`` *after* collapsing rare
      and missing are dropped with a warning (cardinality guard).
    - At transform time, categories unseen during fit are mapped to
      ``__rare__`` (same column as rare training categories).

    Parameters
    ----------
    max_categories:
        Upper bound on one-hot columns emitted per feature (after
        collapsing rare + missing). Columns exceeding this are dropped.
    rare_threshold:
        Minimum training frequency [0, 1) for a category to get its own
        column. Categories strictly below this are collapsed to
        ``__rare__``. Set to 0.0 to disable rare collapsing.

    Example
    -------
    >>> encoder = PolarsOneHotEncoder(max_categories=15, rare_threshold=0.01)
    >>> train = encoder.fit_transform(train_df)
    >>> test  = encoder.transform(test_df)
    """

    max_categories: int = 15
    rare_threshold: float = 0.01

    # col -> ordered list of final categories (including __missing__ / __rare__)
    _categories: dict[str, list[str]] = field(default_factory=dict, init=False)
    # cols dropped at fit time due to exceeding max_categories after collapsing
    _dropped_cols: set[str] = field(default_factory=set, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "PolarsOneHotEncoder":
        """Learn categories and rare/missing buckets from training data."""
        self._categories.clear()
        self._dropped_cols.clear()

        string_cols = [c for c, t in df.schema.items() if t == pl.String]
        n_rows = len(df)

        for col in string_cols:
            series = df[col]

            # --- frequency table including nulls -----------------------
            # null_count gives us the missing count without value_counts quirks
            null_count = series.null_count()
            non_null = series.drop_nulls()

            # After
            counts: dict[str, int] = {
                row[col]: row["count"]
                for row in non_null.value_counts(sort=False).to_dicts()
            }

            # Build final category list: __missing__ first (if any nulls),
            # then non-rare categories in stable alphabetical order,
            # then __rare__ (if any categories are rare).
            final_cats: list[str] = []

            if null_count > 0:
                final_cats.append(_MISSING)

            rare_total = 0
            non_rare: list[str] = []
            for cat, cnt in counts.items():
                freq = cnt / n_rows
                if freq < self.rare_threshold:
                    rare_total += cnt
                else:
                    non_rare.append(cat)

            final_cats.extend(sorted(non_rare))

            # Add __rare__ bucket when at least one category is rare OR
            # when training data had nulls-as-missing (so unseen values
            # at transform time have somewhere to go).
            has_rare_training = rare_total > 0
            if has_rare_training:
                final_cats.append(_RARE)

            if len(final_cats) > self.max_categories:
                print(
                    f"[PolarsOneHotEncoder] WARNING: '{col}' has "
                    f"{len(final_cats)} categories after collapsing "
                    f"(max_categories={self.max_categories}). Column dropped."
                )
                self._dropped_cols.add(col)
                continue

            self._categories[col] = final_cats

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted one-hot encoding."""
        if not self._categories and not self._dropped_cols:
            raise RuntimeError("Call fit() before transform().")

        cols_to_drop: list[str] = []
        all_exprs: list[pl.Expr] = []

        fitted_cols = set(self._categories) | self._dropped_cols
        for col in fitted_cols:
            if col not in df.columns:
                print(
                    f"[PolarsOneHotEncoder] WARNING: fitted column '{col}' "
                    "not found in transform DataFrame. Skipping."
                )
                continue

            if col in self._dropped_cols:
                cols_to_drop.append(col)
                continue

            categories = self._categories[col]
            has_rare = _RARE in categories
            known_non_special = set(categories) - {_MISSING, _RARE}

            for cat in categories:
                new_col = f"{col}_{cat}"

                if cat == _MISSING:
                    # null in input → 1
                    expr = pl.col(col).is_null().cast(pl.Int8).alias(new_col)

                elif cat == _RARE:
                    # rare at train time OR unseen at transform time → 1
                    expr = (
                        pl.col(col)
                        .is_not_null()
                        .__and__(~pl.col(col).is_in(list(known_non_special)))
                        .cast(pl.Int8)
                        .alias(new_col)
                    )

                else:
                    if has_rare:
                        # exact match only; unseen values fall into __rare__
                        expr = (
                            pl.when(pl.col(col).is_null())
                            .then(0)
                            .when(pl.col(col) == cat)
                            .then(1)
                            .otherwise(0)
                            .cast(pl.Int8)
                            .alias(new_col)
                        )
                    else:
                        # no __rare__ bucket: unseen → all zeros (safe default)
                        expr = (
                            pl.when(pl.col(col) == cat)
                            .then(1)
                            .otherwise(0)
                            .cast(pl.Int8)
                            .alias(new_col)
                        )

                all_exprs.append(expr)

            cols_to_drop.append(col)

        if all_exprs:
            df = df.with_columns(all_exprs)

        if cols_to_drop:
            df = df.drop(cols_to_drop)

        return df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def categories(self) -> dict[str, list[str]]:
        """Return the fitted category list per column (read-only copy)."""
        return {col: list(cats) for col, cats in self._categories.items()}

    def feature_names_out(self) -> list[str]:
        """Return all output column names in fit order."""
        return [
            f"{col}_{cat}"
            for col, cats in self._categories.items()
            for cat in cats
        ]