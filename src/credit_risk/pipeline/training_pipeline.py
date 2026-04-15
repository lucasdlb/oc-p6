"""End-to-end training pipeline."""

from __future__ import annotations

from polars import DataFrame

from credit_risk.config import Config, DataSourcesConfig, cfg
from credit_risk.data.cleaner import DataCleaner
from credit_risk.data.encoder import CategoricalEncoder
from credit_risk.data.imputer import DataImputer
from credit_risk.data.loader import PLLazyDataLoader
from credit_risk.data.splitter import DataSplitter
from credit_risk.features.aggregator import FeatureAggregator
from credit_risk.features.transformer import FeatureTransformer
from credit_risk.models.evaluator import ModelEvaluator
from credit_risk.models.predictor import ModelPredictor
from credit_risk.models.trainer import ModelTrainer


class TrainingPipeline:
    def __init__(
        self,
        settings: Config | None = None,
        data_sources: DataSourcesConfig | None = None,
    ):
        self.settings = settings or cfg
        self.data_sources = data_sources or DataSourcesConfig()
        self.loader = PLLazyDataLoader()
        self.cleaner = DataCleaner(self.settings.data, self.data_sources)
        self.imputer = DataImputer(self.settings.data, self.data_sources)
        self.encoder = CategoricalEncoder(self.settings.data)
        self.splitter = DataSplitter(self.settings.data)
        self.aggregator = FeatureAggregator(self.settings.data.features)
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

    def _needs_cleaning(self, table: str) -> bool:
        method = self.data_sources.get_cleaning_method(table)
        return method != "raw"

    def _needs_imputation(self, table: str) -> bool:
        method = self.data_sources.get_imputation_method(table)
        return method != "raw"

    def _clean_table(self, df: DataFrame, table: str) -> DataFrame:
        """Apply cleaning for a specific table."""
        if self._needs_cleaning(table):
            return self.cleaner.clean(df, table)
        return df

    def _impute_table(self, df: DataFrame, table: str) -> DataFrame:
        """Apply imputation for a specific table."""
        if self._needs_imputation(table):
            return self.imputer.impute(df, table)
        return df

    def run(self) -> dict:
        import polars as pl

        labels = self.loader.load_labels()

        app_features = None
        if self.data_sources.is_enabled("application"):
            app_features = self.loader.load_application_features()
            app_features = self._clean_table(app_features, "application")
            app_features = self._impute_table(app_features, "application")

        lazy_data = self.loader.load_all_lazy()

        aggregated_parts = []

        if self.data_sources.is_enabled("bureau"):
            bureau_df = lazy_data["bureau"].collect()
            bureau_clean = self._clean_table(bureau_df, "bureau")
            bureau_clean = self._impute_table(bureau_clean, "bureau")
            bureau_agg = self.aggregator.aggregate_bureau(
                bureau_clean.lazy(), self.settings.data.id_column
            ).collect()
            aggregated_parts.append(("bureau", bureau_agg))

        if self.data_sources.is_enabled("bureau_balance"):
            bb_df = lazy_data["bureau_balance"].collect()
            bb_clean = self._clean_table(bb_df, "bureau_balance")
            bb_clean = self._impute_table(bb_clean, "bureau_balance")
            bb_agg = self.aggregator.aggregate_bureau_balance(
                bb_clean.lazy(),
                self.settings.data.id_column,
            ).collect()
            aggregated_parts.append(("bureau_balance", bb_agg))

        if self.data_sources.is_enabled("previous_application"):
            prev_df = lazy_data["previous_application"].collect()
            prev_clean = self._clean_table(prev_df, "previous_application")
            prev_clean = self._impute_table(prev_clean, "previous_application")
            prev_agg = self.aggregator.aggregate_previous_application(
                prev_clean.lazy(), self.settings.data.id_column
            ).collect()
            aggregated_parts.append(("previous_application", prev_agg))

        if self.data_sources.is_enabled("pos_cash"):
            pos_df = lazy_data["POS_CASH_balance"].collect()
            pos_clean = self._clean_table(pos_df, "pos_cash")
            pos_clean = self._impute_table(pos_clean, "pos_cash")
            pos_agg = self.aggregator.aggregate_POS_CASH(
                pos_clean.lazy(), self.settings.data.id_column
            ).collect()
            aggregated_parts.append(("pos_cash", pos_agg))

        if self.data_sources.is_enabled("installments"):
            ins_df = lazy_data["installments_payments"].collect()
            ins_clean = self._clean_table(ins_df, "installments")
            ins_clean = self._impute_table(ins_clean, "installments")
            ins_agg = self.aggregator.aggregate_installments(
                ins_clean.lazy(), self.settings.data.id_column
            ).collect()
            aggregated_parts.append(("installments", ins_agg))

        if self.data_sources.is_enabled("credit_card"):
            cc_df = lazy_data["credit_card_balance"].collect()
            cc_clean = self._clean_table(cc_df, "credit_card")
            cc_clean = self._impute_table(cc_clean, "credit_card")
            cc_agg = self.aggregator.aggregate_credit_card(
                cc_clean.lazy(), self.settings.data.id_column
            ).collect()
            aggregated_parts.append(("credit_card", cc_agg))

        if app_features is not None:
            main_train = app_features
            if aggregated_parts:
                aggregated_features = aggregated_parts[0][1]
                for _, df in aggregated_parts[1:]:
                    aggregated_features = aggregated_features.join(
                        df, on=self.settings.data.id_column, how="outer_coalesce"
                    )
                main_train = self._merge_features(
                    main_train, aggregated_features, self.settings.data.id_column
                )
            main_train = main_train.join(labels, on=self.settings.data.id_column, how="inner")
        elif aggregated_parts:
            aggregated_features = aggregated_parts[0][1]
            for _, df in aggregated_parts[1:]:
                aggregated_features = aggregated_features.join(
                    df, on=self.settings.data.id_column, how="outer_coalesce"
                )
            main_train = labels.join(
                aggregated_features,
                on=self.settings.data.id_column,
                how="inner",
            )
        else:
            main_train = labels

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

        X_train = train_df.select(feature_cols).to_pandas()
        y_train = train_df.select(target_col).to_numpy().ravel()
        X_val = val_df.select(feature_cols).to_pandas()
        y_val = val_df.select(target_col).to_numpy().ravel()

        X_train = X_train.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)
        X_val = X_val.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

        model = self.trainer.train(X_train, y_train, X_val, y_val, enable_mlflow=False)

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
