"""Prediction script entry point."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from credit_risk.config.settings import get_settings, load_settings
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoder import CategoricalEncoder
from credit_risk.data.loader import DataLoader
from credit_risk.features.aggregator import FeatureAggregator
from credit_risk.features.transformer import FeatureTransformer
from credit_risk.models.predictor import ModelPredictor
from credit_risk.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Predict credit risk")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file path")
    args = parser.parse_args()

    settings = load_settings(args.config)
    logger = setup_logging(settings.logging)
    logger.info("Starting prediction")

    model_path = args.model or str(get_settings().models_path / "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    loader = DataLoader()
    cleaner = DataCleaner(settings.data)
    aggregator = FeatureAggregator(settings.features)
    transformer = FeatureTransformer()
    encoder = CategoricalEncoder(settings.data)
    predictor = ModelPredictor(model)

    main_test = loader.load_application_test()
    main_test = cleaner.clean(main_test)

    lazy_data = loader.load_all_lazy()
    aggregated_features = aggregator.aggregate_all(
        bureau_df=lazy_data["bureau"],
        bureau_balance_df=lazy_data["bureau_balance"],
        prev_app_df=lazy_data["previous_application"],
        pos_df=lazy_data["POS_CASH_balance"],
        installments_df=lazy_data["installments_payments"],
        cc_df=lazy_data["credit_card_balance"],
        id_col=settings.data.id_column,
    )

    main_test = main_test.join(aggregated_features, on=settings.data.id_column, how="left")
    main_test = transformer.transform(main_test)

    cat_cols = encoder.get_low_cardinality_columns(main_test)
    main_test = encoder.label_encode(main_test, cat_cols)

    id_col = settings.data.id_column
    feature_cols = [c for c in main_test.columns if c != id_col]

    X_test = main_test.select(feature_cols).to_numpy()
    X_test = pl.DataFrame(X_test).to_numpy()
    X_test = pl.DataFrame(X_test, schema=feature_cols).to_numpy()
    import numpy as np

    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = predictor.predict_proba(X_test)

    submission = pl.DataFrame(
        {
            settings.data.id_column: main_test.select(id_col).to_series(),
            "TARGET": predictions,
        }
    )
    submission.write_csv(args.output)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
