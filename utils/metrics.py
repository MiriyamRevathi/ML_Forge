"""
MLForge - ML Evaluation Metrics Utility Module
Provides standardized functions to calculate, format, and structure evaluation metrics
for Classification and Regression machine learning models using scikit-learn and numpy.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from config import TASK_CLASSIFICATION, TASK_REGRESSION


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for classification tasks:
    Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(unique_classes) <= 2
    
    # Calculate base metrics
    acc = float(accuracy_score(y_true, y_pred))
    
    average_mode = "binary" if is_binary else "weighted"
    prec = float(precision_score(y_true, y_pred, average=average_mode, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average_mode, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average_mode, zero_division=0))
    
    # ROC-AUC calculation
    roc_auc = None
    if y_prob is not None:
        try:
            if is_binary:
                # For binary, use probability of positive class if 2D array or 1D array
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_positive = y_prob[:, 1]
                else:
                    prob_positive = y_prob
                roc_auc = float(roc_auc_score(y_true, prob_positive))
            else:
                # Multiclass ROC-AUC
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_list = cm.tolist()
    
    class_names = [str(lbl) for lbl in (labels if labels is not None else unique_classes)]

    return {
        "accuracy": round(acc, 4),
        "accuracy_percentage": round(acc * 100, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "is_binary": is_binary,
        "confusion_matrix": {
            "matrix": cm_list,
            "labels": class_names
        }
    }


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for regression tasks:
    MAE, MSE, RMSE, and R2-Score.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "r2_percentage": round(r2 * 100, 2)
    }


def compute_model_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Dispatcher function to compute metrics based on task type.
    """
    if task == TASK_CLASSIFICATION:
        return calculate_classification_metrics(y_true, y_pred, y_prob)
    elif task == TASK_REGRESSION:
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported task type: {task}")
