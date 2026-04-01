"""Model trainer with MLflow tracking."""

import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from credit_risk.config.settings import ModelConfig, get_settings


class ModelTrainer:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self._feature_names: list[str] = []

    def _create_model(self):
        if self.config.model_type == "lightgbm":
            return LGBMClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight="balanced",
                verbose=-1,
            )
        elif self.config.model_type == "sklearn":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_child_samples,
                random_state=self.config.random_state,
                class_weight="balanced",
                n_jobs=self.config.n_jobs,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        experiment_name: str = "credit_risk",
        run_name: str | None = None,
    ) -> LGBMClassifier | RandomForestClassifier:
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.config.model_dump())

            model = self._create_model()
            self._feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

            if X_val is not None and y_val is not None and self.config.model_type == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lambda env: (
                            mlflow.log_metrics(
                                {"val_logloss": env.evaluation_result_list[0][2]},
                                step=env.iteration,
                            )
                            if env.evaluation_result_list
                            else None
                        )
                    ],
                )
            else:
                model.fit(X_train, y_train)

            self.model = model
            mlflow.sklearn.log_model(model, "model")

        return model

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def save(self, path: str | None = None) -> str:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        import pickle

        if path is None:
            path = str(get_settings().models_path / "model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        return path

    def load(self, path: str) -> LGBMClassifier | RandomForestClassifier:
        import pickle

        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self.model
