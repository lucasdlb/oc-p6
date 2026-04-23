"""Model analysis and plotting utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelPlotter:
    """Generate and save model evaluation plots.

    Usage:
        plotter = ModelPlotter()
        plotter.plot_all(y_true, y_pred_proba, threshold=0.5, output_dir="plots")
        plotter.log_to_mlflow(mlflow_active, run_name="model_analysis")
    """

    def __init__(self):
        self.figures: list[tuple[str, plt.Figure]] = []

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_f1_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plot F1 score vs threshold."""
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        ax.plot(thresholds, f1_scores, color="green", lw=2, label="F1 Score")
        ax.axvline(
            x=best_threshold,
            color="red",
            linestyle="--",
            label=f"Best threshold = {best_threshold:.2f} (F1 = {best_f1:.4f})",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score vs Threshold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

        return fig

    def plot_all(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        output_dir: str | Path = "plots",
    ) -> dict[str, float]:
        """Generate all plots and save to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        y_pred = (y_pred_proba >= threshold).astype(int)

        self.plot_roc_curve(y_true, y_pred_proba)
        plt.savefig(output_path / "roc_curve.png", dpi=150, bbox_inches="tight")
        self.figures.append(("roc_curve", plt.gcf()))
        plt.close()

        self.plot_precision_recall_curve(y_true, y_pred_proba)
        plt.savefig(output_path / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
        self.figures.append(("precision_recall_curve", plt.gcf()))
        plt.close()

        self.plot_f1_threshold(y_true, y_pred_proba)
        plt.savefig(output_path / "f1_threshold.png", dpi=150, bbox_inches="tight")
        self.figures.append(("f1_threshold", plt.gcf()))
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        self.plot_roc_curve(y_true, y_pred_proba, axes[0])
        self.plot_precision_recall_curve(y_true, y_pred_proba, axes[1])
        self.plot_f1_threshold(y_true, y_pred_proba, axes[2])

        plt.tight_layout()
        plt.savefig(output_path / "all_curves.png", dpi=150, bbox_inches="tight")
        self.figures.append(("all_curves", plt.gcf()))
        plt.close()

        logger.info(f"Saved plots to {output_path}")

        metrics = {
            "roc_curve_path": str(output_path / "roc_curve.png"),
            "pr_curve_path": str(output_path / "precision_recall_curve.png"),
            "f1_threshold_path": str(output_path / "f1_threshold.png"),
            "combined_path": str(output_path / "all_curves.png"),
        }

        return metrics

    def log_to_mlflow(self, mlflow_active: Any, run_name: str = "model_analysis") -> None:
        """Log all plots as MLflow artifacts."""
        if not mlflow_active:
            logger.info("MLflow not active, skipping plot logging")
            return

        import mlflow

        for name, fig in self.figures:
            fig.savefig(f"/tmp/{name}.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f"/tmp/{name}.png")
            logger.info(f"Logged {name}.png to MLflow")

        plt.close("all")


def analyze_model_results(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    output_dir: str | Path = "plots",
    mlflow_active: Any = None,
) -> dict[str, Any]:
    """Convenience function to analyze model and generate all plots.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Threshold for binary predictions
        output_dir: Directory to save plots
        mlflow_active: MLflow active run (if any)

    Returns:
        Dictionary with metrics and plot paths
    """
    plotter = ModelPlotter()
    metrics = plotter.plot_all(y_true, y_pred_proba, threshold, output_dir)
    plotter.log_to_mlflow(mlflow_active)

    return metrics
