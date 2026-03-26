"""Reports generation for model interpretability."""

from pathlib import Path
from typing import Any

import numpy as np


class InterpretabilityReport:
    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or [f"feature_{i}" for i in range(100)]

    def global_importance_report(
        self, shap_values: np.ndarray, mean_abs_importance: np.ndarray
    ) -> list[dict[str, Any]]:
        sorted_idx = np.argsort(mean_abs_importance)[::-1]
        report = []
        for idx in sorted_idx[:20]:
            report.append(
                {
                    "feature": self.feature_names[idx]
                    if idx < len(self.feature_names)
                    else f"feature_{idx}",
                    "mean_abs_shap": float(mean_abs_importance[idx]),
                    "rank": len(report) + 1,
                }
            )
        return report

    def local_importance_report(
        self,
        shap_values: np.ndarray,
        base_values: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        contributions = shap_values
        ranked = sorted(enumerate(contributions), key=lambda x: abs(x[1]), reverse=True)[:10]

        return {
            "base_value": float(base_values[0]) if len(base_values) > 0 else 0.0,
            "prediction": float(shap_values.sum())
            + float(base_values[0] if len(base_values) > 0 else 0),
            "top_features": [
                {
                    "feature": feature_names[idx]
                    if feature_names and idx < len(feature_names)
                    else f"feature_{idx}",
                    "contribution": float(val),
                }
                for idx, val in ranked
            ],
        }

    def save_report(self, report: dict[str, Any], path: Path) -> None:
        import json

        with open(path, "w") as f:
            json.dump(report, f, indent=2)
