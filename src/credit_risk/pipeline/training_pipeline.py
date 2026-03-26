"""End-to-end training pipeline."""

import numpy as np
from polars import DataFrame

from credit_risk.config.settings import Settings
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoder import CategoricalEncoder
from credit_risk.data.loader import DataLoader
from credit_risk.data.splitter import DataSplitter
from credit_risk.features.aggregator import FeatureAggregator
from credit_risk.features.transformer import FeatureTransformer
from credit_risk.models.evaluator import ModelEvaluator
from credit_risk.models.predictor import ModelPredictor
from credit_risk.models.trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.loader = DataLoader(self.settings.data)
        self.cleaner = DataCleaner(self.settings.data)
        self.encoder = CategoricalEncoder(self.settings.data)
        self.splitter = DataSplitter(self.settings.data)
        self.aggregator = FeatureAggregator(self.settings.features)
        self.transformer = FeatureTransformer()
        self.trainer = ModelTrainer(self.settings.model)
        self.evaluator = ModelEvaluator()

    def _merge_features(
        self,
        main_df: DataFrame,
        aggregated_df: DataFrame,
        id_col: str,
    ) -> DataFrame:
        return main_df.join(aggregated_df, on=id_col, how="left")

    def run(self) -> dict:
        import polars as pl

        main_train = self.loader.load_application_train()
        main_train = self.cleaner.clean(main_train)

        lazy_data = self.loader.load_all_lazy()
        aggregated_features = self.aggregator.aggregate_all(
            bureau_df=lazy_data["bureau"],
            bureau_balance_df=lazy_data["bureau_balance"],
            prev_app_df=lazy_data["previous_application"],
            pos_df=lazy_data["POS_CASH_balance"],
            installments_df=lazy_data["installments_payments"],
            cc_df=lazy_data["credit_card_balance"],
            id_col=self.settings.data.id_column,
        )

        main_train = self._merge_features(
            main_train, aggregated_features, self.settings.data.id_column
        )
        main_train = self.transformer.transform(main_train)

        string_cols = self.encoder.get_categorical_columns(main_train)
        main_train = self.encoder.label_encode(main_train, string_cols)

        target_col = self.settings.data.target_column
        id_col = self.settings.data.id_column

        non_numeric_cols = [c for c in main_train.columns if main_train.schema[c] == pl.String]
        if non_numeric_cols:
            main_train = main_train.drop(non_numeric_cols)

        train_df, val_df = self.splitter.split(main_train)

        feature_cols = self.encoder.get_feature_names(main_train, target_col, id_col)

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(target_col).to_numpy().ravel()
        X_val = val_df.select(feature_cols).to_numpy()
        y_val = val_df.select(target_col).to_numpy().ravel()

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        model = self.trainer.train(X_train, y_train, X_val, y_val)

        predictor = ModelPredictor(model)
        y_pred = predictor.predict(X_val)
        y_pred_proba = predictor.predict_proba(X_val)

        metrics = self.evaluator.evaluate(y_val, y_pred, y_pred_proba)

        return {
            "model": model,
            "metrics": metrics,
            "feature_names": feature_cols,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
        }
