"""NaNReplacer — sklearn transformer that replaces NaN and inf values in numpy arrays."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NanFillStrategy = float | Literal["median_mode"]


class NaNReplacer(TransformerMixin, BaseEstimator):
    """Replace NaN and inf values using a per-column or constant strategy.

    Two strategies are supported, controlled by ``fill_value``:

    - **Constant** (``fill_value`` is a ``float``): replaces every NaN/inf
      with the given scalar.  ``fit`` is a no-op; the replacement is
      stateless.
    - **Median/mode** (``fill_value="median_mode"``): stateful imputer.
      ``fit`` learns, for each column, the median (numeric) or the most
      frequent value (non-numeric / object).  ``transform`` applies those
      per-column fill values and also replaces ±inf with the same value.

    In both strategies the fitted fill values are stored in
    ``fill_values_`` (a 1-D numpy array of ``float64``, length = number of
    columns seen at fit time).

    Args:
        fill_value: Scalar constant or ``"median_mode"`` strategy selector.
    """

    def __init__(self, fill_value: NanFillStrategy = 0.0):
        self.fill_value = fill_value

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NaNReplacer:
        """Learn per-column fill values from training data.

        Args:
            X: 2-D numeric array of shape (n_samples, n_features).
            y: Ignored; present for sklearn API compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        if self.fill_value == "median_mode":
            fills = np.empty(n_features, dtype=float)
            for j in range(n_features):
                col = X[:, j]
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    fills[j] = 0.0
                else:
                    fills[j] = float(np.median(finite))
        else:
            fills = np.full(n_features, float(self.fill_value))

        self.fill_values_: np.ndarray = fills
        self.n_features_in_: int = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaN and ±inf values using fitted fill values.

        Args:
            X: 2-D numeric array of shape (n_samples, n_features).

        Returns:
            Array with the same shape as ``X`` with NaN/inf replaced.
        """
        check_is_fitted(self, "fill_values_")
        X = np.array(X, dtype=float)

        for j, fill in enumerate(self.fill_values_):
            col = X[:, j]
            mask = ~np.isfinite(col)
            if mask.any():
                col[mask] = fill

        return X
