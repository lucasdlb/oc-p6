"""MLflow utility class - encapsulates logging without conditional checks."""

from __future__ import annotations

import json
import os
import tempfile

import mlflow

from credit_risk.config.config import CONFIG_DIR
from credit_risk.config.config_grid import ConfigGrid


class MlflowLogger:
    """MLflow logging helper - instantiate and use methods."""

    def log_params(self, params: dict) -> None:
        """Log params, converting all values to strings."""
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_param(self, key: str, value) -> None:
        """Log a single param."""
        mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics one-by-one."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_metric(self, key: str, value, step: int | None = None) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def start_run(self, run_name: str, **kwargs):
        """Smart run opener - auto-detects parent and sets nested accordingly."""
        parent = mlflow.active_run()
        return mlflow.start_run(run_name=run_name, nested=bool(parent), **kwargs)

    def log_config(self, config_obj) -> None:
        """Log any dataclass as flattened params."""
        from dataclasses import asdict

        self.log_params(asdict(config_obj))

    def log_dict_artifact(
        self, data: dict, filename: str, artifact_path: str = "artifacts"
    ) -> None:
        """Save dict as JSON artifact."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
        os.unlink(tmp_path)

    def log_file_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        """Log a file as an artifact."""
        mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def log_model(self, model, name: str) -> None:
        """Log sklearn model."""
        mlflow.sklearn.log_model(model, name)

    def log_grid_config(self, grid: "ConfigGrid") -> None:
        """Log grid axes and base config with only varying values as params."""
        # Log base config with single values (not lists)
        # base_config_dict = self._dict_with_single_values(grid._raw)
        # self.log_params(base_config_dict)

        # Log grid axes as separate params
        for axis_key, values in grid.axes.items():
            # Convert dots to underscores for mlflow param naming
            param_name = axis_key.replace(".", "__")
            # Store as JSON string since mlflow params need to be strings
            import json

            self.log_param(param_name, json.dumps(values))

    def _dict_with_single_values(self, d: dict) -> dict:
        """Convert single-element lists to scalars, keep multi-element lists as lists."""

        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._dict_with_single_values(v)
            elif isinstance(v, list):
                # If it's a single element list, extract the value
                if len(v) == 1:
                    result[k] = v[0]
                else:
                    # Keep multi-element lists as is (they'll be logged in log_grid_config)
                    result[k] = v
            else:
                result[k] = v
        return result

    def log_flat_config(self, cfg, exclude_keys: set | None = None) -> None:
        """Log all config as flattened params, excluding static keys from data.toml."""
        import tomllib

        exclude = exclude_keys or set()
        data_toml_path = CONFIG_DIR / "data.toml"
        if data_toml_path.exists():
            with open(data_toml_path, "rb") as f:
                data_toml_keys = set(tomllib.load(f).keys())
                exclude.update(data_toml_keys)

        flat_params = {}
        for key, value in cfg.model_dump().items():
            if key in exclude:
                continue
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_params[f"{key}_{subkey}"] = str(subvalue)
            else:
                flat_params[key] = str(value)

        mlflow.log_params(flat_params)
