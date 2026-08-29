"""
MLForge ML Engine - Ensemble Classification Architecture Catalogue Module
Provides specialized wrappers and hyperparameter schemas for Random Forest,
Extra Trees, Gradient Boosting, AdaBoost, and HistGradientBoosting classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)


class EnsembleClassifierCatalogue:
    """
    Ensemble methods classification algorithms suite.
    """

    @staticmethod
    def create_random_forest(params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
        params = params or {}
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else None,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", True)),
            criterion=str(params.get("criterion", "gini")),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )

    @staticmethod
    def create_extra_trees(params: Optional[Dict[str, Any]] = None) -> ExtraTreesClassifier:
        params = params or {}
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else None,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", False)),
            criterion=str(params.get("criterion", "gini")),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1
        )

    @staticmethod
    def create_gradient_boosting(params: Optional[Dict[str, Any]] = None) -> GradientBoostingClassifier:
        params = params or {}
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            subsample=float(params.get("subsample", 1.0)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_adaboost(params: Optional[Dict[str, Any]] = None) -> AdaBoostClassifier:
        params = params or {}
        return AdaBoostClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            algorithm=str(params.get("algorithm", "SAMME")),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_hist_gradient_boosting(params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingClassifier:
        params = params or {}
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") is not None else None,
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            random_state=int(params.get("random_state", 42))
        )
