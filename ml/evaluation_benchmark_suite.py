"""
MLForge ML Engine - Evaluation Benchmark Suite Module
Calculates classification (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix)
and regression (MAE, MSE, RMSE, R², MAPE, MedAE, EVR) metrics, per-class breakdown, and ranking matrices.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ml.classification_metrics import ClassificationMetricsCalculator
from ml.regression_metrics import RegressionMetricsCalculator


class EvaluationBenchmarkSuite:
    """
    Comprehensive ML Model Evaluation & Metric Benchmarking Suite.
    """

    @staticmethod
    def evaluate_model_performance(
        task_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_labels: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates performance evaluation metrics based on task type.
        """
        if task_type == "classification":
            return ClassificationMetricsCalculator.compute_all_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                class_labels=class_labels
            )
        else:
            return RegressionMetricsCalculator.compute_all_metrics(
                y_true=y_true,
                y_pred=y_pred
            )

    @staticmethod
    def rank_models_by_performance(
        models_metrics: List[Dict[str, Any]],
        task_type: str = "classification"
    ) -> List[Dict[str, Any]]:
        """
        Ranks multiple models based on primary evaluation metric.
        """
        ranked = list(models_metrics)
        primary = "accuracy" if task_type == "classification" else "r2_score"

        if task_type == "classification":
            ranked.sort(key=lambda m: m.get("metrics", {}).get("accuracy", 0.0), reverse=True)
        else:
            ranked.sort(key=lambda m: m.get("metrics", {}).get("r2_score", -999.0), reverse=True)

        for i, item in enumerate(ranked, start=1):
            item["rank"] = i
            item["primary_metric"] = primary
            item["primary_value"] = item.get("metrics", {}).get(primary, 0.0)

        return ranked
