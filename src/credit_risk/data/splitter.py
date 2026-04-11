"""Train/validation/test splitter."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from polars import DataFrame

from credit_risk.config.settings import DataConfig

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(
        self,
        config: DataConfig | None = None,
        random_state: int = 42,
    ):
        self.config = config or DataConfig()
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def split(self, df: DataFrame, stratify: bool = True) -> tuple[DataFrame, DataFrame]:
        logger.info(f"Splitting dataframe with {df.height} rows (stratify={stratify})")
        train_size = self.config.train_size
        target_col = self.config.target_column

        if stratify:
            target_counts = df.group_by(target_col).len()
            train_ratio = train_size

            train_dfs = []
            val_dfs = []

            for target_val in target_counts[target_col]:
                subset = df.filter(pl.col(target_col) == target_val)
                n_train = int(len(subset) * train_ratio)
                indices = self._rng.permutation(len(subset))
                train_idx = indices[:n_train]
                val_idx = indices[n_train:]
                train_dfs.append(subset[train_idx])
                val_dfs.append(subset[val_idx])

            train_df = pl.concat(train_dfs)
            val_df = pl.concat(val_dfs)
        else:
            n = len(df)
            indices = self._rng.permutation(n)
            n_train = int(n * train_size)
            train_df = df[indices[:n_train]]
            val_df = df[indices[n_train:]]

        logger.info(f"Split complete: train={train_df.height}, val={val_df.height}")
        return train_df, val_df
