"""
MLForge ML Engine - Cross-Validation Module
Executes K-Fold and Stratified K-Fold cross-validation for Classification and Regression,
computing mean performance metrics, standard deviation bounds, and per-fold scores.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate


class CrossValidationEngine:
    """
    K-Fold & Stratified K-Fold cross-validation engine.
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
        """
        Executes N-Fold cross-validation and returns detailed fold metrics.
        """
        if task_type == "classification":
            scoring = {
                "acc": "accuracy",
                "prec": "precision_weighted",
                "rec": "recall_weighted",
                "f1": "f1_weighted"
            }
            # Use Stratified K-Fold if possible
            unique_counts = np.bincount(y.astype(int)) if np.issubdtype(y.dtype, np.integer) else []
            if len(unique_counts) > 0 and (unique_counts >= n_splits).all():
                cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            else:
                cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error"
            }
            cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        fold_scores = {}
        mean_metrics = {}
        std_metrics = {}

        if task_type == "classification":
            for metric_key, score_name in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "f1_score")]:
                raw_scores = cv_results[f"test_{metric_key}"]
                fold_scores[score_name] = [round(float(s), 4) for s in raw_scores]
                mean_metrics[score_name] = round(float(np.mean(raw_scores)), 4)
                std_metrics[score_name] = round(float(np.std(raw_scores)), 4)
        else:
            # Regression metrics
            r2_scores = cv_results["test_r2"]
            mae_scores = -cv_results["test_mae"]
            mse_scores = -cv_results["test_mse"]
            rmse_scores = np.sqrt(mse_scores)

            fold_scores["r2"] = [round(float(s), 4) for s in r2_scores]
            fold_scores["mae"] = [round(float(s), 4) for s in mae_scores]
            fold_scores["rmse"] = [round(float(s), 4) for s in rmse_scores]

            mean_metrics["r2"] = round(float(np.mean(r2_scores)), 4)
            mean_metrics["mae"] = round(float(np.mean(mae_scores)), 4)
            mean_metrics["rmse"] = round(float(np.mean(rmse_scores)), 4)

            std_metrics["r2"] = round(float(np.std(r2_scores)), 4)
            std_metrics["mae"] = round(float(np.std(mae_scores)), 4)
            std_metrics["rmse"] = round(float(np.std(rmse_scores)), 4)

        return {
            "n_splits": n_splits,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "fold_scores": fold_scores,
            "fit_time_seconds": round(float(np.mean(cv_results["fit_time"])), 3)
        }
