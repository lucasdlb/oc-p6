"""Prediction script entry point."""

import argparse
import pickle

import numpy as np
import polars as pl

from credit_risk.config import load_config
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.models.predictor import ModelPredictor
from credit_risk.pipeline.processing_pipeline import ProcessingPipeline
from credit_risk.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Predict credit risk")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file path")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging()
    logger.info("Starting prediction")

    model_path = args.model or "models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    loader = PLLazyDataLoader()
    predictor = ModelPredictor(model)

    id_col = cfg.data.target.id_column

    main_test = loader.load("application_test").collect()
    main_test = ProcessingPipeline(cfg.data.application).fit_transform(main_test)

    feature_cols = [c for c in main_test.columns if c != id_col]

    X_test = main_test.select(feature_cols).to_numpy()
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = predictor.predict_proba(X_test)

    submission = pl.DataFrame(
        {
            id_col: main_test.select(id_col).to_series(),
            "TARGET": predictions,
        }
    )
    submission.write_csv(args.output)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
