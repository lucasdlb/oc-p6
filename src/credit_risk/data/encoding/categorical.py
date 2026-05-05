"""Stateful categorical encoder for string columns."""

from __future__ import annotations

import logging
from typing import override

import polars as pl

from credit_risk.data.base import ProcessingStep

logger = logging.getLogger(__name__)


class CategoricalEncoder(ProcessingStep):
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

    def __init__(self, high_cardinality_threshold: int = 10) -> None:
        self.high_cardinality_threshold = high_cardinality_threshold
        self._vocabularies: dict[str, dict[str, int]] = {}
        self._freq_maps: dict[str, dict[str, float]] = {}
        self._strategies: dict[str, str] = {}
        self.is_fitted_: bool = False

    @override
    def fit(self, X: pl.DataFrame, y=None) -> "CategoricalEncoder":
        """Learn vocabularies from X. Call only on training data."""
        string_cols = [c for c, t in X.schema.items() if t == pl.String]

        if len(string_cols) == 0:
            logger.warning("CategoricalEncoder: No categorical columns detected.")
            self.is_fitted_ = True

            return self

        for col in string_cols:
            series = X[col].drop_nulls()
            unique = sorted(series.unique().to_list())
            n_unique = len(unique)

            if n_unique <= 2:
                self._vocabularies[col] = {v: i for i, v in enumerate(unique)}
                self._strategies[col] = "binary"

            elif n_unique <= self.high_cardinality_threshold:
                self._vocabularies[col] = {v: i for i, v in enumerate(unique)}
                self._strategies[col] = "label"

            else:
                counts = X[col].value_counts(sort=True)
                total = len(X)
                self._freq_maps[col] = {row[col]: row["count"] / total for row in counts.to_dicts()}
                self._strategies[col] = "frequency"

        self.is_fitted_ = True

        return self

    @override
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Apply fitted encoding. Unknown categories → -1 (label) or 0.0 (freq)."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")

        if not self._strategies:
            return X

        exprs = []
        for col, strategy in self._strategies.items():
            if col not in X.columns:
                continue

            if strategy in ("binary", "label"):
                vocab = self._vocabularies[col]
                values = [vocab.get(v, -1) if v is not None else -1 for v in X[col].to_list()]
                mapping_series = pl.Series(name=col, values=values, dtype=pl.Int16)
                exprs.append(mapping_series)

            elif strategy == "frequency":
                freq_map = self._freq_maps[col]
                values = [
                    freq_map.get(v, 0.0) if v is not None else float("nan")
                    for v in X[col].to_list()
                ]
                freq_series = pl.Series(name=col, values=values, dtype=pl.Float32)
                exprs.append(freq_series)

        return X.with_columns(exprs)

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
