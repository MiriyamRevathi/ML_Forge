"""
MLForge ML Engine - Hyperparameter Tuning & Grid Search Module
Implements Grid Search (GridSearchCV) and Randomized Search (RandomizedSearchCV)
for hyperparameter tuning and model optimization.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ml.models.classification import ClassificationCatalogue
from ml.models.regression import RegressionCatalogue


class HyperparameterTuner:
    """
    GridSearch and RandomizedSearch hyperparameter optimizer.
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
        """
        Executes Grid Search cross-validation over parameter space.
        """
        if task_type == "classification":
            base_model = ClassificationCatalogue.create_classifier(model_key)
            default_scoring = scoring or "accuracy"
        else:
            base_model = RegressionCatalogue.create_regressor(model_key)
            default_scoring = scoring or "r2"

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=default_scoring,
            n_jobs=-1,
            return_train_score=False
        )

        grid_search.fit(X_train, y_train)

        return {
            "best_params": grid_search.best_params_,
            "best_score": round(float(grid_search.best_score_), 4),
            "scoring_metric": default_scoring,
            "best_estimator": grid_search.best_estimator_,
            "total_candidates": len(grid_search.cv_results_["params"])
        }

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
        """
        Executes Randomized Search cross-validation over parameter distribution.
        """
        if task_type == "classification":
            base_model = ClassificationCatalogue.create_classifier(model_key)
            default_scoring = scoring or "accuracy"
        else:
            base_model = RegressionCatalogue.create_regressor(model_key)
            default_scoring = scoring or "r2"

        rand_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=default_scoring,
            random_state=random_state,
            n_jobs=-1
        )

        rand_search.fit(X_train, y_train)

        return {
            "best_params": rand_search.best_params_,
            "best_score": round(float(rand_search.best_score_), 4),
            "scoring_metric": default_scoring,
            "best_estimator": rand_search.best_estimator_,
            "total_iterations": n_iter
        }
