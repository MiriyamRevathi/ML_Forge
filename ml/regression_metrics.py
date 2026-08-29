"""
MLForge ML Engine - Detailed Regression Metrics Calculator Module
Calculates MAE, MSE, RMSE, R² Score, MAPE, Median Absolute Error,
Explained Variance Ratio, and Residual Analysis Statistics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)


class RegressionMetricsCalculator:
    """
    Comprehensive Regression Metrics Evaluator.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes complete suite of regression metrics and residual statistics.
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            mape = None

        medae = float(median_absolute_error(y_true, y_pred))
        evr = float(explained_variance_score(y_true, y_pred))

        # Residual Statistics
        residuals = y_true - y_pred
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        res_max = float(np.max(np.abs(residuals)))

        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "r2_percentage": round(r2 * 100, 2),
            "mape": round(mape, 4) if mape is not None else None,
            "median_absolute_error": round(medae, 4),
            "explained_variance_score": round(evr, 4),
            "residual_stats": {
                "residual_mean": round(res_mean, 4),
                "residual_std": round(res_std, 4),
                "max_residual": round(res_max, 4)
            }
        }
