"""InferencePipeline — complete inference chain (processing + model prediction).

Wraps fitted per-table processing pipelines, a fitted model pipeline, and
the selected feature list into a single serializable object.  At inference
time uses transform() (not fit_transform()) — no re-fitting.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from credit_risk_models.estimator_pipeline import EstimatorPipeline

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Complete inference chain — data processing + model prediction.

    Usage::

        # Training side — fit and save
        inference = InferencePipeline(
            processing_pipelines=fitted_processing,  # dict[str, Pipeline]
            model_pipeline=final_model,              # EstimatorPipeline
            feature_names=selected_features,         # list[str]
        )
        inference.save("inference.pkl")

        # Inference side — load and predict
        inference = InferencePipeline.load("inference.pkl")
        ids, probas = inference.predict(raw_tables)
    """

    def __init__(
        self,
        processing_pipelines: dict[str, Pipeline],
        model_pipeline: EstimatorPipeline,
        feature_names: list[str],
        id_column: str = "SK_ID_CURR",
        cross_transformer: Any = None,
    ) -> None:
        """Initialize with fitted pipelines and model.

        Args:
            processing_pipelines: Per-table fitted sklearn Pipelines
                (table_name → fitted pipeline).
            model_pipeline: Fitted EstimatorPipeline.
            feature_names: Ordered list of feature names the model expects.
            id_column: Join key column name (must match processing step schemas).
            cross_transformer: Optional cross-table transformer (stateless).
        """
        self.processing_pipelines = processing_pipelines
        self.model_pipeline = model_pipeline
        self.feature_names = feature_names
        self.id_column = id_column
        self.cross_transformer = cross_transformer

    def predict(self, raw_tables: dict[str, pl.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        """Transform raw tables → join → cross features → filter → predict.

        Args:
            raw_tables: Mapping of table name → raw Polars DataFrame.
                Tables with no corresponding pipeline are silently skipped.

        Returns:
            Tuple of (ids, probabilities) as 1-D numpy arrays.
        """
        processed: dict[str, pl.DataFrame] = {}

        for name, pipeline in self.processing_pipelines.items():
            if name not in raw_tables:
                continue
            out = pipeline.transform(raw_tables[name])
            processed[name] = self._prefix_columns(out, name)

        if not processed:
            raise ValueError(
                "No tables were processed. Check that raw_tables keys "
                "match processing_pipelines keys."
            )

        table_names = list(processed)
        merged = processed[table_names[0]]
        for name in table_names[1:]:
            merged = merged.join(processed[name], on=self.id_column, how="left")

        if self.cross_transformer is not None:
            from credit_risk_processing.data.base import NoOpStep

            if not isinstance(self.cross_transformer, NoOpStep):
                if hasattr(self.cross_transformer, "id_column"):
                    self.cross_transformer.id_column = self.id_column
                cross_out = self.cross_transformer.transform(merged)
                cross_df = cross_out.get("cross")
                if cross_df is not None:
                    cross_cols = [c for c in cross_df.columns if c != self.id_column]
                    if cross_cols:
                        merged = merged.join(
                            cross_df.select([self.id_column] + cross_cols),
                            on=self.id_column,
                            how="left",
                        )

        ids = merged.select(self.id_column).to_series().to_numpy()

        available = set(merged.columns)
        cols = [f for f in self.feature_names if f in available]
        missing = [f for f in self.feature_names if f not in available]
        if missing:
            logger.warning(
                "%d features from training not found in inference data: %s",
                len(missing),
                missing,
            )
        logger.info("Using %d / %d features", len(cols), len(self.feature_names))

        X = merged.select(cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probas = self.model_pipeline.predict_proba(X)

        return ids, probas

    def _prefix_columns(self, df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """Prefix all columns except id_column with table name."""
        rename_map = {}
        for col in df.columns:
            if col != self.id_column and not col.startswith(f"{prefix}_"):
                rename_map[col] = f"{prefix}_{col}"
        return df.rename(rename_map) if rename_map else df

    def save(self, path: Path) -> None:
        """Serialize to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("InferencePipeline saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> InferencePipeline:
        """Deserialize from a pickle file.

        Args:
            path: Path to the pickled InferencePipeline.

        Returns:
            Deserialized InferencePipeline instance.

        Raises:
            TypeError: If the unpickled object is not an InferencePipeline.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, InferencePipeline):
            raise TypeError(f"Expected InferencePipeline, got {type(obj).__name__}")
        return obj
