"""
MLForge ML Engine - Linear Regression Architecture Catalogue Module
Provides specialized wrappers, hyperparameter schemas, fitting methods,
and coefficient feature importance extractors for Linear Regression, Ridge, Lasso,
ElasticNet, Huber, and Bayesian Ridge regressors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, SGDRegressor


class LinearRegressorCatalogue:
    """
    Linear and regularized regression algorithms suite.
    """

    @staticmethod
    def get_ridge_schema() -> Dict[str, Any]:
        return {
            "alpha": {
                "type": "float",
                "default": 1.0,
                "min": 0.001,
                "max": 100.0,
                "description": "Regularization strength; larger values specify stronger regularization."
            },
            "solver": {
                "type": "choice",
                "default": "auto",
                "options": ["auto", "svd", "cholesky", "lsqr", "sag"],
                "description": "Solver calculation algorithm."
            }
        }

    @staticmethod
    def get_lasso_schema() -> Dict[str, Any]:
        return {
            "alpha": {
                "type": "float",
                "default": 1.0,
                "min": 0.001,
                "max": 100.0,
                "description": "Constant multiplier of L1 penalty term."
            },
            "max_iter": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 5000,
                "description": "Maximum iterations."
            }
        }

    @staticmethod
    def create_linear_regression(params: Optional[Dict[str, Any]] = None) -> LinearRegression:
        params = params or {}
        return LinearRegression(
            fit_intercept=bool(params.get("fit_intercept", True))
        )

    @staticmethod
    def create_ridge(params: Optional[Dict[str, Any]] = None) -> Ridge:
        params = params or {}
        return Ridge(
            alpha=float(params.get("alpha", 1.0)),
            solver=str(params.get("solver", "auto")),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_lasso(params: Optional[Dict[str, Any]] = None) -> Lasso:
        params = params or {}
        return Lasso(
            alpha=float(params.get("alpha", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_elastic_net(params: Optional[Dict[str, Any]] = None) -> ElasticNet:
        params = params or {}
        return ElasticNet(
            alpha=float(params.get("alpha", 1.0)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=int(params.get("random_state", 42))
        )

    @staticmethod
    def create_huber(params: Optional[Dict[str, Any]] = None) -> HuberRegressor:
        params = params or {}
        return HuberRegressor(
            epsilon=float(params.get("epsilon", 1.35)),
            max_iter=int(params.get("max_iter", 500)),
            alpha=float(params.get("alpha", 0.0001))
        )

    @staticmethod
    def create_bayesian_ridge(params: Optional[Dict[str, Any]] = None) -> BayesianRidge:
        params = params or {}
        return BayesianRidge(
            n_iter=int(params.get("n_iter", 300)),
            alpha_1=float(params.get("alpha_1", 1e-6)),
            lambda_1=float(params.get("lambda_1", 1e-6))
        )

    @staticmethod
    def create_sgd_regressor(params: Optional[Dict[str, Any]] = None) -> SGDRegressor:
        params = params or {}
        return SGDRegressor(
            loss=str(params.get("loss", "squared_error")),
            penalty=str(params.get("penalty", "l2")),
            alpha=float(params.get("alpha", 0.0001)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=int(params.get("random_state", 42))
        )
