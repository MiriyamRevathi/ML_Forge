"""
MLForge ML Engine - Boosting Classification Architecture Catalogue Module
Provides specialized wrappers, hyperparameter schemas, fitting methods,
class probability calculations, and feature importance extractors for Gradient Boosting,
AdaBoost, and HistGradientBoosting Classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier


class BoostingClassifierCatalogue:
    """
    Gradient and Adaptive Boosting classification algorithms suite.
    """

    @staticmethod
    def get_gradient_boosting_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of boosting stages to perform."
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "min": 0.001,
                "max": 1.0,
                "description": "Learning rate shrinks contribution of each tree."
            },
            "max_depth": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 20,
                "description": "Maximum depth of individual regression estimators."
            },
            "subsample": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 1.0,
                "description": "Fraction of samples used for fitting individual base learners."
            }
        }

    @staticmethod
    def get_adaboost_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 50,
                "min": 10,
                "max": 500,
                "description": "Maximum number of estimators at which boosting is terminated."
            },
            "learning_rate": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 2.0,
                "description": "Weight applied to each classifier at each boosting iteration."
            }
        }

    @staticmethod
    def create_gradient_boosting(params: Optional[Dict[str, Any]] = None) -> GradientBoostingClassifier:
        params = params or {}
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 1.0)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_adaboost(params: Optional[Dict[str, Any]] = None) -> AdaBoostClassifier:
        params = params or {}
        return AdaBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_hist_gradient_boosting(params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingClassifier:
        params = params or {}
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(params.get("random_state", 42))
        )
