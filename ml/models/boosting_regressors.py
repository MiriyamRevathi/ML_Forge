"""
MLForge ML Engine - Boosting Regression Architecture Catalogue Module
Provides specialized wrappers, hyperparameter schemas, fitting methods,
and feature importance extractors for Gradient Boosting Regressor, AdaBoost Regressor,
and HistGradientBoosting Regressor.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor


class BoostingRegressorCatalogue:
    """
    Gradient and Adaptive Boosting regression algorithms suite.
    """

    @staticmethod
    def get_gradient_boosting_regressor_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of boosting stages."
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "min": 0.001,
                "max": 1.0,
                "description": "Learning rate shrinks step size."
            },
            "max_depth": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 20,
                "description": "Max tree depth."
            },
            "loss": {
                "type": "choice",
                "default": "squared_error",
                "options": ["squared_error", "absolute_error", "huber", "quantile"],
                "description": "Loss function to optimize."
            }
        }

    @staticmethod
    def create_gradient_boosting_regressor(params: Optional[Dict[str, Any]] = None) -> GradientBoostingRegressor:
        params = params or {}
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            loss=str(params.get("loss", "squared_error")),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_adaboost_regressor(params: Optional[Dict[str, Any]] = None) -> AdaBoostRegressor:
        params = params or {}
        return AdaBoostRegressor(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            loss=str(params.get("loss", "linear")),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_hist_gradient_boosting_regressor(params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingRegressor:
        params = params or {}
        return HistGradientBoostingRegressor(
            max_iter=int(params.get("max_iter", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(params.get("random_state", 42))
        )
