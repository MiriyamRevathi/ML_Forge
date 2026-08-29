"""
MLForge ML Engine - Cross Validation Suite Module
Executes K-Fold and Stratified K-Fold cross-validation for Classification and Regression,
computing mean performance metrics, standard deviation bounds, and per-fold scores.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.cross_validation import CrossValidationEngine


class CrossValidationSuite:
    """
    K-Fold & Stratified K-Fold cross-validation engine suite.
    """

    @staticmethod
    def run_cross_validation(
        model: Any,
        task_type: str,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        return CrossValidationEngine.run_cross_validation(
            model=model,
            task_type=task_type,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state
        )
