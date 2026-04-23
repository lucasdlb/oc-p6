"""Threshold selection for classification using pre-computed CV predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

__all__ = [
    "ThresholdSelector",
    "SimpleThresholdSelector",
    "ThresholdSelectorResult",
    "create_cost_sensitive_score",
]


class ThresholdSelector:
    """Protocol for threshold selection from CV predictions."""

    def select(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Select optimal threshold.

        Args:
            y_true: True labels
            y_proba: Positive-class predicted probabilities

        Returns:
            Optimal threshold value
        """
        ...


class SimpleThresholdSelector:
    """Select optimal classification threshold by optimizing a metric over thresholds.

    This is a stateless utility: it takes pre-computed (y_true, y_proba) from a CV run
    and finds the threshold that maximizes (or minimizes) the given metric.

    Usage:
        selector = SimpleThresholdSelector(metric="f1")
        threshold = selector.select(y_true, y_proba)

        # With custom scorer
        selector = SimpleThresholdSelector(custom_func=my_cost_func)
        threshold = selector.select(y_true, y_proba)
    """

    def __init__(
        self,
        metric: str = "f1",
        direction: str = "maximize",
        thresholds: list[float] | None = None,
        custom_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        """Initialize SimpleThresholdSelector.

        Args:
            metric: Built-in metric to optimize ("f1", "precision", "recall")
            direction: "maximize" or "minimize"
            thresholds: Custom list of thresholds to try
            custom_func: Custom function that takes (y_true, y_pred_binary) and returns a score
        """
        self.metric = metric
        self.direction = direction
        self.thresholds = thresholds or np.arange(0.01, 0.99, 0.01).tolist()
        self._custom_func = custom_func

    def _get_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute score based on metric or custom function."""
        if self._custom_func is not None:
            return self._custom_func(y_true, y_pred)

        if self.metric == "f1":
            return f1_score(y_true, y_pred, zero_division=0)
        elif self.metric == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        elif self.metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def select(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Select optimal threshold by optimizing a metric.

        Args:
            y_true: True labels
            y_proba: Positive-class predicted probabilities

        Returns:
            Optimal threshold value
        """
        best_threshold = 0.5
        best_score = -1.0 if self.direction == "maximize" else float("inf")

        for threshold in self.thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = self._get_score(y_true, y_pred)

            if self.direction == "maximize":
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            else:
                if score < best_score:
                    best_score = score
                    best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.2f} ({self.metric}={best_score:.4f})")
        return best_threshold


@dataclass
class ThresholdSelectorResult:
    """Result from threshold selection."""

    optimal_threshold: float
    best_score: float
    metric: str


def create_cost_sensitive_score(
    fn_weight: float = 1.0,
    fp_weight: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create custom cost-sensitive scoring function.

    Args:
        fn_weight: Weight for false negatives (missed defaults)
        fp_weight: Weight for false positives (false alarms)

    Returns:
        Function that returns HIGHER score for LOWER cost (use direction="maximize")
    """

    def cost_sensitive_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Higher score = lower cost. Higher fn_weight = penalize misses more."""
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        cost = fp * fp_weight + fn * fn_weight

        return -cost  # Negative so higher = lower cost

    return cost_sensitive_score
