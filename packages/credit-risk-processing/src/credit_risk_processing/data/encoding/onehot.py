"""One-hot encoder using Polars native operations."""

from __future__ import annotations

import logging
from typing import override

import polars as pl
from sklearn.preprocessing import TargetEncoder

from credit_risk_processing.data.base import ProcessingStep

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
            col for col, dtype in X.schema.items() if dtype in (pl.String, pl.Categorical, pl.Utf8)
        ]

    def fit(self, X: pl.DataFrame, y=None) -> "PolarsTargetEncoder":
        self.cat_cols_ = self._detect_cat_cols(X)

        if len(self.cat_cols_) == 0:
            logger.warning("PolarsTargetEncoder: No categorical columns detected.")
            self.is_fitted_ = True

            return self

        # Only fit on categorical columns
        self.model.fit(X.select(self.cat_cols_), y)
        self.is_fitted_ = True

        return self

    def transform(self, X: pl.DataFrame, y=None):
        if not self.is_fitted_:
            raise RuntimeError("Encoder not fitted")

        if len(self.cat_cols_) == 0:
            return X

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
    - When the number of categories after collapsing still exceeds
      ``max_categories``, the least frequent categories beyond the top
      ``max_categories - 1`` are also collapsed into ``__rare__``.
      The column is **never dropped** — all information is preserved
      via the ``__rare__`` bucket.
    - At transform time, categories unseen during fit are mapped to
      ``__rare__`` (same column as rare training categories).

    Parameters
    ----------
    max_categories:
        Maximum number of one-hot columns emitted per feature (including
        ``__missing__`` and ``__rare__``).  Excess categories are folded
        into ``__rare__`` rather than dropping the column.
    rare_threshold:
        Minimum training frequency [0, 1) for a category to get its own
        column. Categories strictly below this are collapsed to
        ``__rare__``. Set to 0.0 to disable frequency-based collapsing.
    """

    def __init__(self, max_categories: int = 15, rare_threshold: float = 0.01) -> None:
        self.max_categories = max_categories
        self.rare_threshold = rare_threshold
        self.categories_: dict[str, list[str]] = {}
        self.is_fitted: bool = False

    @override
    def fit(self, X: pl.DataFrame, y=None) -> "PolarsOneHotEncoder":
        """Learn categories and rare/missing buckets from training data."""
        self.categories_.clear()

        string_cols = [c for c, t in X.schema.items() if t == pl.String]
        n_rows = len(X)

        for col in string_cols:
            series = X[col]

            null_count = series.null_count()
            non_null = series.drop_nulls()

            # Sort by descending frequency so we can trim the tail deterministically.
            counts: list[tuple[str, int]] = sorted(
                ((row[col], row["count"]) for row in non_null.value_counts(sort=False).to_dicts()),
                key=lambda x: -x[1],
            )

            final_cats: list[str] = []

            if null_count > 0:
                final_cats.append(_MISSING)

            # Split into rare (below threshold) and non-rare, preserving freq order.
            non_rare: list[str] = []
            has_rare = False
            for cat, cnt in counts:
                if cnt / n_rows < self.rare_threshold:
                    has_rare = True
                else:
                    non_rare.append(cat)

            final_cats.extend(non_rare)
            if has_rare:
                final_cats.append(_RARE)

            # If still over max_categories, trim the least-frequent non-special
            # categories into __rare__ rather than dropping the column.
            if len(final_cats) > self.max_categories:
                # How many non-special slots do we have?
                # Reserve 1 slot for __rare__ (will always exist after trimming).
                n_special = (1 if _MISSING in final_cats else 0) + 1  # +1 for __rare__
                n_keep = self.max_categories - n_special
                non_special = [c for c in final_cats if c not in (_MISSING, _RARE)]

                kept = non_special[:n_keep]
                trimmed = non_special[n_keep:]

                rebuilt: list[str] = []
                if _MISSING in final_cats:
                    rebuilt.append(_MISSING)
                rebuilt.extend(kept)
                if trimmed or has_rare:
                    rebuilt.append(_RARE)

                logger.debug(
                    "PolarsOneHotEncoder: '%s' had %d categories; trimmed %d into __rare__ "
                    "(max_categories=%d).",
                    col,
                    len(final_cats),
                    len(trimmed),
                    self.max_categories,
                )
                final_cats = rebuilt

            self.categories_[col] = final_cats

        self.is_fitted = True

        return self

    @override
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Apply fitted one-hot encoding."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        if not self.categories_:
            return X

        cols_to_drop: list[str] = []
        all_exprs: list[pl.Expr] = []

        for col, categories in self.categories_.items():
            if col not in X.columns:
                logger.warning(
                    "PolarsOneHotEncoder: fitted column '%s' not found in transform "
                    "DataFrame. Skipping.",
                    col,
                )
                continue

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
