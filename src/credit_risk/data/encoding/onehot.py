"""One-hot encoder using Polars native operations."""

from __future__ import annotations

import logging

import polars as pl

from typing import override

from narwhals import DataFrame
from sklearn.preprocessing import TargetEncoder

from credit_risk.data.base import ProcessingStep

logger = logging.getLogger(__name__)

_MISSING = "__missing__"
_RARE = "__rare__"


class PolarsTargetEncoder(ProcessingStep):
    def __init__(self) -> None:
        self.model: TargetEncoder = TargetEncoder().set_output(transform="polars")
        self.is_fitted_: bool = False
        self.cat_cols_: list[str] = []

    def _detect_cat_cols(self, X: pl.DataFrame) -> list[str]:
        """Detect categorical columns robustly (not only pl.String)."""
        return [
            col for col, dtype in X.schema.items()
            if dtype in (pl.String, pl.Categorical, pl.Utf8)
        ]

    def fit(self, X: pl.DataFrame, y=None) -> "PolarsTargetEncoder":
        self.cat_cols_ = self._detect_cat_cols(X)

        # Only fit on categorical columns
        self.model.fit(X.select(self.cat_cols_), y)

        self.is_fitted_ = True
        return self

    def transform(self, X: pl.DataFrame, y=None):
        if not self.is_fitted_:
            raise RuntimeError("Encoder not fitted")

        X_cat = X.select(self.cat_cols_)
        encoded = self.model.transform(X_cat)

        # Replace only categorical columns, keep numeric untouched
        return X.drop(self.cat_cols_).hstack(encoded)

    def __sklearn_is_fitted__(self):
        return self.is_fitted_


class PolarsOneHotEncoder(ProcessingStep):
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
    """

    def __init__(self, max_categories: int = 15, rare_threshold: float = 0.01) -> None:
        self.max_categories = max_categories
        self.rare_threshold = rare_threshold
        self.categories_: dict[str, list[str]] = {}
        self.dropped_cols_: set[str] = set()

    @override
    def fit(self, X: pl.DataFrame, y=None) -> "PolarsOneHotEncoder":
        """Learn categories and rare/missing buckets from training data."""
        self.categories_.clear()
        self.dropped_cols_.clear()

        string_cols = [c for c, t in X.schema.items() if t == pl.String]
        n_rows = len(X)

        for col in string_cols:
            series = X[col]

            null_count = series.null_count()
            non_null = series.drop_nulls()

            counts: dict[str, int] = {
                row[col]: row["count"] for row in non_null.value_counts(sort=False).to_dicts()
            }

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

            has_rare_training = rare_total > 0
            if has_rare_training:
                final_cats.append(_RARE)

            if len(final_cats) > self.max_categories:
                logger.warning(
                    "PolarsOneHotEncoder: '%s' has %d categories after collapsing "
                    "(max_categories=%d). Column dropped.",
                    col,
                    len(final_cats),
                    self.max_categories,
                )
                self.dropped_cols_.add(col)
                continue

            self.categories_[col] = final_cats

        return self

    @override
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Apply fitted one-hot encoding."""
        if not self.categories_ and not self.dropped_cols_:
            raise RuntimeError("Call fit() before transform().")

        cols_to_drop: list[str] = []
        all_exprs: list[pl.Expr] = []

        fitted_cols = set(self.categories_) | self.dropped_cols_
        for col in fitted_cols:
            if col not in X.columns:
                logger.warning(
                    "PolarsOneHotEncoder: fitted column '%s' not found in transform "
                    "DataFrame. Skipping.",
                    col,
                )
                continue

            if col in self.dropped_cols_:
                cols_to_drop.append(col)
                continue

            categories = self.categories_[col]
            has_rare = _RARE in categories
            known_non_special = set(categories) - {_MISSING, _RARE}

            for cat in categories:
                new_col = f"{col}_{cat}"

                if cat == _MISSING:
                    expr = pl.col(col).is_null().cast(pl.Int8).alias(new_col)

                elif cat == _RARE:
                    expr = (
                        pl.col(col)
                        .is_not_null()
                        .__and__(~pl.col(col).is_in(list(known_non_special)))
                        .cast(pl.Int8)
                        .alias(new_col)
                    )

                else:
                    if has_rare:
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
            X = X.with_columns(all_exprs)

        if cols_to_drop:
            X = X.drop(cols_to_drop)

        return X

    def categories(self) -> dict[str, list[str]]:
        """Return the fitted category list per column (read-only copy)."""
        return {col: list(cats) for col, cats in self.categories_.items()}

    def feature_names_out(self) -> list[str]:
        """Return all output column names in fit order."""
        return [f"{col}_{cat}" for col, cats in self.categories_.items() for cat in cats]
