"""
MLForge ML Engine - Regression Metrics Suite Module
Calculates MAE, MSE, RMSE, R² Score, MAPE, Median Absolute Error,
Explained Variance Ratio, and Residual Analysis Statistics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ml.regression_metrics import RegressionMetricsCalculator


class RegressionMetricsSuite:
    """
    Comprehensive Regression Metrics Evaluator Suite.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        return RegressionMetricsCalculator.compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred
        )
