"""
MLForge ML Engine - Classification Metrics Suite Module
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
Confusion Matrix, Log Loss, and Per-Class Breakdown Metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ml.classification_metrics import ClassificationMetricsCalculator


class ClassificationMetricsSuite:
    """
    Comprehensive Classification Metrics Evaluator Suite.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_labels: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        return ClassificationMetricsCalculator.compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_labels=class_labels
        )
