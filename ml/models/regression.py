"""
MLForge ML Engine - Production Regression Model Catalogue Module
Provides scikit-learn wrappers, hyperparameter schemas, fitting methods,
and feature importance extractors for 10 regression algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)


class RegressionCatalogue:
    """
    Catalogue and factory for 10 production Regression algorithms.
    """

    SUPPORTED_REGRESSORS = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "elastic_net": "ElasticNet Regression",
        "decision_tree_regressor": "Decision Tree Regressor",
        "random_forest_regressor": "Random Forest Regressor",
        "extra_trees_regressor": "Extra Trees Regressor",
        "gradient_boosting_regressor": "Gradient Boosting Regressor",
        "adaboost_regressor": "AdaBoost Regressor",
        "hist_gradient_boosting_regressor": "HistGradientBoosting Regressor"
    }

    @staticmethod
    def get_hyperparameter_schema(model_key: str) -> Dict[str, Any]:
        """
        Returns hyperparameter names, types, defaults, and valid options for UI form generation.
        """
        key = model_key.lower().replace(" ", "_")
        
        if key == "linear_regression":
            return {
                "fit_intercept": {"type": "bool", "default": True, "description": "Whether to calculate intercept for this model"}
            }
        elif key == "ridge":
            return {
                "alpha": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "description": "L2 Regularization strength"},
                "solver": {"type": "choice", "default": "auto", "options": ["auto", "svd", "cholesky", "lsqr", "sag"], "description": "Solver calculation algorithm"}
            }
        elif key == "lasso":
            return {
                "alpha": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "description": "L1 Regularization penalty"},
                "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000, "description": "Maximum iterations"}
            }
        elif key == "elastic_net":
            return {
                "alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "description": "Penalty multiplier"},
                "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "description": "ElasticNet mixing parameter (0=L2, 1=L1)"}
            }
        elif key == "decision_tree_regressor":
            return {
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Maximum tree depth"},
                "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Minimum samples required to split node"}
            }
        elif key == "random_forest_regressor":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Number of trees"},
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Max tree depth"},
                "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Min split samples"}
            }
        elif key == "extra_trees_regressor":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Number of trees"},
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Max tree depth"}
            }
        elif key == "gradient_boosting_regressor":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Boosting stages"},
                "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Learning rate"},
                "max_depth": {"type": "int", "default": 3, "min": 1, "max": 10, "description": "Max tree depth"}
            }
        elif key == "adaboost_regressor":
            return {
                "n_estimators": {"type": "int", "default": 50, "min": 10, "max": 300, "description": "Max estimators"},
                "learning_rate": {"type": "float", "default": 1.0, "min": 0.01, "max": 2.0, "description": "Learning rate"}
            }
        elif key == "hist_gradient_boosting_regressor":
            return {
                "max_iter": {"type": "int", "default": 100, "min": 10, "max": 300, "description": "Max iterations"},
                "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Learning rate"}
            }
        else:
            return {}

    @staticmethod
    def create_regressor(model_key: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Instantiates scikit-learn regressor object with validated hyperparameter payload.
        """
        params = params or {}
        key = model_key.lower().replace(" ", "_")
        rs = int(params.get("random_state", 42))

        if key == "linear_regression":
            return LinearRegression(fit_intercept=bool(params.get("fit_intercept", True)))
        elif key == "ridge":
            return Ridge(
                alpha=float(params.get("alpha", 1.0)),
                solver=str(params.get("solver", "auto")),
                random_state=rs
            )
        elif key == "lasso":
            return Lasso(
                alpha=float(params.get("alpha", 1.0)),
                max_iter=int(params.get("max_iter", 1000)),
                random_state=rs
            )
        elif key == "elastic_net":
            return ElasticNet(
                alpha=float(params.get("alpha", 1.0)),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                random_state=rs
            )
        elif key == "decision_tree_regressor":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return DecisionTreeRegressor(
                max_depth=depth,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=rs
            )
        elif key == "random_forest_regressor":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=depth,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=rs,
                n_jobs=-1
            )
        elif key == "extra_trees_regressor":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return ExtraTreesRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=depth,
                random_state=rs,
                n_jobs=-1
            )
        elif key == "gradient_boosting_regressor":
            return GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=rs
            )
        elif key == "adaboost_regressor":
            return AdaBoostRegressor(
                n_estimators=int(params.get("n_estimators", 50)),
                learning_rate=float(params.get("learning_rate", 1.0)),
                random_state=rs
            )
        elif key == "hist_gradient_boosting_regressor":
            return HistGradientBoostingRegressor(
                max_iter=int(params.get("max_iter", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=rs
            )
        else:
            return RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=-1)

    @staticmethod
    def extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extracts feature importances or linear regression coefficients.
        """
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef)

        if importances is not None and len(importances) == len(feature_names):
            total = float(np.sum(importances))
            norm_imp = importances / total if total > 0 else importances
            imp_dict = {feat: round(float(val), 4) for feat, val in zip(feature_names, norm_imp)}
            return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

        return None
