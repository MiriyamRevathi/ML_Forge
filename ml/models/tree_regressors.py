"""
MLForge ML Engine - Tree-Based Regression Architecture Catalogue Module
Provides specialized wrappers, hyperparameter validation schemas, fitting methods,
and feature importance extractors for Decision Tree, Random Forest, and Extra Trees Regressors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


class TreeRegressorCatalogue:
    """
    Tree-based decision regression algorithms suite.
    """

    @staticmethod
    def get_decision_tree_regressor_schema() -> Dict[str, Any]:
        return {
            "criterion": {
                "type": "choice",
                "default": "squared_error",
                "options": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                "description": "Function to measure split quality."
            },
            "max_depth": {
                "type": "int_nullable",
                "default": None,
                "min": 1,
                "max": 50,
                "description": "Maximum tree depth."
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Min samples required to split internal node."
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 20,
                "description": "Min samples required at leaf node."
            }
        }

    @staticmethod
    def get_random_forest_regressor_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of trees in ensemble."
            },
            "max_depth": {
                "type": "int_nullable",
                "default": None,
                "min": 1,
                "max": 100,
                "description": "Max tree depth."
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Min samples required to split."
            },
            "bootstrap": {
                "type": "bool",
                "default": True,
                "description": "Whether bootstrap samples are used."
            }
        }

    @staticmethod
    def create_decision_tree_regressor(params: Optional[Dict[str, Any]] = None) -> DecisionTreeRegressor:
        params = params or {}
        depth = int(params["max_depth"]) if params.get("max_depth") is not None else None

        return DecisionTreeRegressor(
            criterion=str(params.get("criterion", "squared_error")),
            max_depth=depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_random_forest_regressor(params: Optional[Dict[str, Any]] = None) -> RandomForestRegressor:
        params = params or {}
        depth = int(params["max_depth"]) if params.get("max_depth") is not None else None

        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )

    @staticmethod
    def create_extra_trees_regressor(params: Optional[Dict[str, Any]] = None) -> ExtraTreesRegressor:
        params = params or {}
        depth = int(params["max_depth"]) if params.get("max_depth") is not None else None

        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            bootstrap=bool(params.get("bootstrap", False)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )
