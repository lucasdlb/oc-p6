"""Classification metrics for cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "ClassificationRankingMetrics",
    "ClassificationMetrics",
]


@dataclass
class ClassificationRankingMetrics:
    """Threshold-independent classification metrics.

    Always computes: roc_auc, pr_auc, gini, log_loss, average_precision, brier_score, ks.

    Usage:
        metrics = ClassificationRankingMetrics()
        scores = metrics.compute(y_true, y_proba)
    """

    def compute(self, y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
        """Compute all ranking metrics.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class

        Returns:
            Dictionary of metric_name -> score
        """
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))

        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
        ks = float(np.max(tpr_curve - fpr_curve))

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "gini": float(2 * roc_auc - 1),
            "log_loss": float(log_loss(y_true, y_proba)),
            "average_precision": pr_auc,
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "ks": ks,
        }

    def names(self) -> list[str]:
        """Return list of metric names."""
        return ["roc_auc", "pr_auc", "gini", "log_loss", "average_precision", "brier_score", "ks"]


@dataclass
class ClassificationMetrics:
    """Composite metrics: ranking + threshold-dependent.

    Wraps ClassificationRankingMetrics and adds threshold-based metrics
    (f1, recall, precision, accuracy, specificity, npv, balanced_accuracy, mcc,
    confusion matrix counts) computed at a given threshold.

    Usage:
        metrics = ClassificationMetrics(
            ranking=ClassificationRankingMetrics(),
            f1=True,
            recall=True,
            threshold=0.5,
        )
        scores = metrics.compute(y_true, y_proba)
    """

    ranking: ClassificationRankingMetrics | None = None
    f1: bool = False
    recall: bool = False
    precision: bool = False
    accuracy: bool = False
    specificity: bool = False
    npv: bool = False
    balanced_accuracy: bool = False
    mcc: bool = False
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.ranking is None:
            self.ranking = ClassificationRankingMetrics()

    def compute(self, y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
        """Compute all metrics.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class

        Returns:
            Dictionary of metric_name -> score
        """
        assert self.ranking is not None
        scores = self.ranking.compute(y_true, y_proba)

        y_binary = (y_proba >= self.threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_binary, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        scores["tp"] = float(tp)
        scores["fp"] = float(fp)
        scores["tn"] = float(tn)
        scores["fn"] = float(fn)
        scores["tpr"] = tpr
        scores["tnr"] = tnr
        scores["fpr"] = fpr
        scores["fnr"] = fnr
        scores["ppv"] = ppv
        scores["npv"] = npv_val

        if self.f1:
            scores["f1"] = float(f1_score(y_true, y_binary, zero_division=0))
        if self.recall:
            scores["recall"] = float(recall_score(y_true, y_binary, zero_division=0))
        if self.precision:
            scores["precision"] = float(precision_score(y_true, y_binary, zero_division=0))
        if self.accuracy:
            scores["accuracy"] = float(accuracy_score(y_true, y_binary))
        if self.specificity:
            scores["specificity"] = tnr
        if self.npv:
            scores["npv"] = npv_val
        if self.balanced_accuracy:
            scores["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_binary))
        if self.mcc:
            scores["mcc"] = float(matthews_corrcoef(y_true, y_binary))

        return scores

    def names(self) -> list[str]:
        """Return list of all enabled metric names."""
        assert self.ranking is not None
        ranking_names = self.ranking.names()
        threshold_names = [
            n for n in ("f1", "recall", "precision", "accuracy", "specificity", "npv", "balanced_accuracy", "mcc")
            if getattr(self, n)
        ]
        return ranking_names + threshold_names