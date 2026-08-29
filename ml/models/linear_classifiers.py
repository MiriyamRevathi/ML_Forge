"""
MLForge ML Engine - Linear & Kernel Classification Architecture Catalogue Module
Provides specialized wrappers, hyperparameter schemas, fitting methods,
and probability calculations for Logistic Regression, Support Vector Classifier (SVC),
and Gaussian Naive Bayes classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class LinearClassifierCatalogue:
    """
    Linear, kernel, and probabilistic classification algorithms suite.
    """

    @staticmethod
    def get_logistic_regression_schema() -> Dict[str, Any]:
        return {
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.001,
                "max": 100.0,
                "description": "Inverse regularization strength; smaller values specify stronger regularization."
            },
            "penalty": {
                "type": "choice",
                "default": "l2",
                "options": ["l2", "l1", "elasticnet", "none"],
                "description": "Regularization norm penalty."
            },
            "solver": {
                "type": "choice",
                "default": "lbfgs",
                "options": ["lbfgs", "saga", "liblinear"],
                "description": "Algorithm used in optimization problem."
            },
            "max_iter": {
                "type": "int",
                "default": 500,
                "min": 100,
                "max": 5000,
                "description": "Maximum iterations for solver convergence."
            }
        }

    @staticmethod
    def get_svc_schema() -> Dict[str, Any]:
        return {
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "description": "Regularization parameter."
            },
            "kernel": {
                "type": "choice",
                "default": "rbf",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "description": "Kernel type used in algorithm."
            },
            "degree": {
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 5,
                "description": "Degree of polynomial kernel function ('poly')."
            },
            "gamma": {
                "type": "choice",
                "default": "scale",
                "options": ["scale", "auto"],
                "description": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'."
            }
        }

    @staticmethod
    def create_logistic_regression(params: Optional[Dict[str, Any]] = None) -> LogisticRegression:
        params = params or {}
        penalty = str(params.get("penalty", "l2"))
        solver = str(params.get("solver", "lbfgs"))

        # Adjust solver for penalty constraints
        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            solver = "saga"
        elif penalty == "elasticnet" and solver != "saga":
            solver = "saga"

        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=penalty if penalty != "none" else None,
            solver=solver,
            max_iter=int(params.get("max_iter", 500)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_svc(params: Optional[Dict[str, Any]] = None) -> SVC:
        params = params or {}
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "rbf")),
            degree=int(params.get("degree", 3)),
            gamma=str(params.get("gamma", "scale")),
            probability=True,
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_naive_bayes(params: Optional[Dict[str, Any]] = None) -> GaussianNB:
        params = params or {}
        return GaussianNB(
            var_smoothing=float(params.get("var_smoothing", 1e-9))
        )

    @staticmethod
    def create_ridge_classifier(params: Optional[Dict[str, Any]] = None) -> RidgeClassifier:
        params = params or {}
        return RidgeClassifier(
            alpha=float(params.get("alpha", 1.0)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_sgd_classifier(params: Optional[Dict[str, Any]] = None) -> SGDClassifier:
        params = params or {}
        return SGDClassifier(
            loss=str(params.get("loss", "log_loss")),
            penalty=str(params.get("penalty", "l2")),
            alpha=float(params.get("alpha", 0.0001)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=int(params.get("random_state", 42))
        )
