"""Train/validation/test splitter."""

import numpy as np
import polars as pl
from polars import DataFrame

from credit_risk.config.settings import DataConfig


class DataSplitter:
    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()

    def split(self, df: DataFrame, stratify: bool = True) -> tuple[DataFrame, DataFrame]:
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
                indices = np.random.permutation(len(subset))
                train_idx = indices[:n_train]
                val_idx = indices[n_train:]
                train_dfs.append(subset[train_idx])
                val_dfs.append(subset[val_idx])

            train_df = pl.concat(train_dfs)
            val_df = pl.concat(val_dfs)
        else:
            n = len(df)
            indices = np.random.permutation(n)
            n_train = int(n * train_size)
            train_df = df[indices[:n_train]]
            val_df = df[indices[n_train:]]

        return train_df, val_df
