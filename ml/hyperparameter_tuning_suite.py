"""
MLForge ML Engine - Hyperparameter Tuning Suite Module
Implements Grid Search (GridSearchCV) and Randomized Search (RandomizedSearchCV)
for hyperparameter tuning and model optimization.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.hyperparameter_tuning import HyperparameterTuner


class HyperparameterTuningSuite:
    """
    GridSearch and RandomizedSearch hyperparameter optimizer suite.
    """

    @staticmethod
    def run_grid_search(
        task_type: str,
        model_key: str,
        param_grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        return HyperparameterTuner.run_grid_search(
            task_type=task_type,
            model_key=model_key,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring=scoring
        )

    @staticmethod
    def run_randomized_search(
        task_type: str,
        model_key: str,
        param_distributions: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 10,
        cv: int = 3,
        scoring: Optional[str] = None,
        random_state: int = 42
    ) -> Dict[str, Any]:
        return HyperparameterTuner.run_randomized_search(
            task_type=task_type,
            model_key=model_key,
            param_distributions=param_distributions,
            X_train=X_train,
            y_train=y_train,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state
        )
