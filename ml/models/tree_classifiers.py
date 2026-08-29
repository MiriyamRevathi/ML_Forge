"""
MLForge ML Engine - Tree-Based Classification Architecture Catalogue Module
Provides specialized wrappers, hyperparameter validation schemas, fitting methods,
class probability calculations, and feature importance extractors for Decision Tree,
Random Forest, and Extra Trees Classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class TreeClassifierCatalogue:
    """
    Tree-based decision algorithms suite.
    """

    @staticmethod
    def get_decision_tree_schema() -> Dict[str, Any]:
        return {
            "criterion": {
                "type": "choice",
                "default": "gini",
                "options": ["gini", "entropy", "log_loss"],
                "description": "Function to measure the quality of a split."
            },
            "splitter": {
                "type": "choice",
                "default": "best",
                "options": ["best", "random"],
                "description": "Strategy used to choose split at each node."
            },
            "max_depth": {
                "type": "int_nullable",
                "default": None,
                "min": 1,
                "max": 50,
                "description": "Maximum depth of the tree."
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 50,
                "description": "Minimum samples required to split an internal node."
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 50,
                "description": "Minimum samples required to be at a leaf node."
            },
            "max_features": {
                "type": "choice",
                "default": "sqrt",
                "options": ["sqrt", "log2", "none"],
                "description": "Number of features to consider when looking for best split."
            }
        }

    @staticmethod
    def get_random_forest_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of decision trees in the forest ensemble."
            },
            "criterion": {
                "type": "choice",
                "default": "gini",
                "options": ["gini", "entropy", "log_loss"],
                "description": "Function to measure split quality."
            },
            "max_depth": {
                "type": "int_nullable",
                "default": None,
                "min": 1,
                "max": 100,
                "description": "Maximum depth of trees."
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Minimum samples required to split."
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 20,
                "description": "Minimum samples required at a leaf node."
            },
            "bootstrap": {
                "type": "bool",
                "default": True,
                "description": "Whether bootstrap samples are used when building trees."
            }
        }

    @staticmethod
    def get_extra_trees_schema() -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of trees in Extra Trees ensemble."
            },
            "max_depth": {
                "type": "int_nullable",
                "default": None,
                "min": 1,
                "max": 100,
                "description": "Maximum depth of trees."
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 20,
                "description": "Min split samples."
            },
            "bootstrap": {
                "type": "bool",
                "default": False,
                "description": "Whether to use bootstrap samples."
            }
        }

    @staticmethod
    def create_decision_tree(params: Optional[Dict[str, Any]] = None) -> DecisionTreeClassifier:
        params = params or {}
        max_depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
        max_feat = params.get("max_features")
        if max_feat == "none":
            max_feat = None

        return DecisionTreeClassifier(
            criterion=str(params.get("criterion", "gini")),
            splitter=str(params.get("splitter", "best")),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_feat,
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_random_forest(params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
        params = params or {}
        max_depth = int(params["max_depth"]) if params.get("max_depth") is not None else None

        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            criterion=str(params.get("criterion", "gini")),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )

    @staticmethod
    def create_extra_trees(params: Optional[Dict[str, Any]] = None) -> ExtraTreesClassifier:
        params = params or {}
        max_depth = int(params["max_depth"]) if params.get("max_depth") is not None else None

        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            bootstrap=bool(params.get("bootstrap", False)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )
