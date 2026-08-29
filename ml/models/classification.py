"""
MLForge ML Engine - Production Classification Model Catalogue Module
Provides scikit-learn wrappers, hyperparameter schemas, fitting methods,
class probability calculations, and feature importance extractors for 10 classification algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class ClassificationCatalogue:
    """
    Catalogue and factory for 10 production Classification algorithms.
    """

    SUPPORTED_CLASSIFIERS = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree Classifier",
        "random_forest": "Random Forest Classifier",
        "extra_trees": "Extra Trees Classifier",
        "knn": "K-Nearest Neighbors",
        "svm": "Support Vector Classifier (SVC)",
        "naive_bayes": "Gaussian Naive Bayes",
        "gradient_boosting": "Gradient Boosting Classifier",
        "adaboost": "AdaBoost Classifier",
        "hist_gradient_boosting": "HistGradientBoosting Classifier"
    }

    @staticmethod
    def get_hyperparameter_schema(model_key: str) -> Dict[str, Any]:
        """
        Returns hyperparameter names, types, defaults, and valid options for UI form generation.
        """
        key = model_key.lower().replace(" ", "_")
        
        if key == "logistic_regression":
            return {
                "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0, "description": "Inverse regularization strength"},
                "penalty": {"type": "choice", "default": "l2", "options": ["l2", "none"], "description": "Regularization norm"},
                "max_iter": {"type": "int", "default": 500, "min": 100, "max": 2000, "description": "Maximum iterations for solver to converge"}
            }
        elif key == "decision_tree":
            return {
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Maximum depth of the tree"},
                "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Minimum samples required to split a node"},
                "criterion": {"type": "choice", "default": "gini", "options": ["gini", "entropy", "log_loss"], "description": "Function to measure split quality"}
            }
        elif key == "random_forest":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Number of trees in the forest"},
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Maximum depth of trees"},
                "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "description": "Minimum samples required to split"},
                "criterion": {"type": "choice", "default": "gini", "options": ["gini", "entropy"], "description": "Split measurement criterion"}
            }
        elif key == "extra_trees":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Number of trees"},
                "max_depth": {"type": "int_nullable", "default": None, "min": 1, "max": 50, "description": "Max tree depth"}
            }
        elif key == "knn":
            return {
                "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 30, "description": "Number of neighbors"},
                "weights": {"type": "choice", "default": "uniform", "options": ["uniform", "distance"], "description": "Weight function"}
            }
        elif key == "svm":
            return {
                "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 50.0, "description": "Regularization parameter"},
                "kernel": {"type": "choice", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"], "description": "Kernel type"}
            }
        elif key == "naive_bayes":
            return {
                "var_smoothing": {"type": "float", "default": 1e-9, "min": 1e-12, "max": 1e-5, "description": "Variance smoothing portion"}
            }
        elif key == "gradient_boosting":
            return {
                "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Boosting stages"},
                "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Learning rate shrinks step size"},
                "max_depth": {"type": "int", "default": 3, "min": 1, "max": 10, "description": "Max depth of individual trees"}
            }
        elif key == "adaboost":
            return {
                "n_estimators": {"type": "int", "default": 50, "min": 10, "max": 300, "description": "Maximum estimators"},
                "learning_rate": {"type": "float", "default": 1.0, "min": 0.01, "max": 2.0, "description": "Weight applied to each classifier"}
            }
        elif key == "hist_gradient_boosting":
            return {
                "max_iter": {"type": "int", "default": 100, "min": 10, "max": 300, "description": "Maximum boosting iterations"},
                "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Learning rate"}
            }
        else:
            return {}

    @staticmethod
    def create_classifier(model_key: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Instantiates scikit-learn classifier object with validated hyperparameter payload.
        """
        params = params or {}
        key = model_key.lower().replace(" ", "_")
        rs = int(params.get("random_state", 42))

        if key == "logistic_regression":
            penalty = str(params.get("penalty", "l2"))
            solver = "lbfgs" if penalty in ["l2", "none"] else "saga"
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                penalty=penalty if penalty != "none" else None,
                solver=solver,
                max_iter=int(params.get("max_iter", 500)),
                random_state=rs
            )
        elif key == "decision_tree":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=int(params.get("min_samples_split", 2)),
                criterion=str(params.get("criterion", "gini")),
                random_state=rs
            )
        elif key == "random_forest":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=depth,
                min_samples_split=int(params.get("min_samples_split", 2)),
                criterion=str(params.get("criterion", "gini")),
                random_state=rs,
                n_jobs=-1
            )
        elif key == "extra_trees":
            depth = int(params["max_depth"]) if params.get("max_depth") is not None else None
            return ExtraTreesClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=depth,
                random_state=rs,
                n_jobs=-1
            )
        elif key == "knn":
            return KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 5)),
                weights=str(params.get("weights", "uniform")),
                n_jobs=-1
            )
        elif key == "svm":
            return SVC(
                C=float(params.get("C", 1.0)),
                kernel=str(params.get("kernel", "rbf")),
                probability=True,
                random_state=rs
            )
        elif key == "naive_bayes":
            return GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9)))
        elif key == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=rs
            )
        elif key == "adaboost":
            return AdaBoostClassifier(
                n_estimators=int(params.get("n_estimators", 50)),
                learning_rate=float(params.get("learning_rate", 1.0)),
                random_state=rs
            )
        elif key == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=int(params.get("max_iter", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                random_state=rs
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1)

    @staticmethod
    def extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extracts normalized feature importances or linear coefficients if supported by model architecture.
        """
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim == 2 else np.abs(coef)

        if importances is not None and len(importances) == len(feature_names):
            total = float(np.sum(importances))
            norm_imp = importances / total if total > 0 else importances
            imp_dict = {feat: round(float(val), 4) for feat, val in zip(feature_names, norm_imp)}
            # Sort descending
            return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

        return None
