"""
MLForge ML Engine - Evaluation Module
Evaluates trained models on test dataset, computes metrics, and generates predictions/probabilities.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from utils.metrics import compute_model_metrics


class ModelEvaluator:
    """
    Model evaluation engine.
    """

    @staticmethod
    def evaluate(
        model: Any,
        task_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
        """
        Executes prediction on test data and computes metric evaluation dictionary.
        """
        y_pred = model.predict(X_test)
        
        y_prob = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None
                
        metrics = compute_model_metrics(
            task=task_type,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        return metrics, y_pred, y_prob
