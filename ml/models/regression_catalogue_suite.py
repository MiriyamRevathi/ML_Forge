"""
MLForge ML Engine - Extended Regression Catalogue Suite Module
Provides scikit-learn wrappers, hyperparameter schemas, fitting methods,
and feature importance extractors for 10 regression algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from ml.models.regression import RegressionCatalogue


class ExtendedRegressionCatalogueSuite:
    """
    Catalogue and factory for 10 production Regression algorithms.
    """

    SUPPORTED_REGRESSORS = RegressionCatalogue.SUPPORTED_REGRESSORS

    @staticmethod
    def get_hyperparameter_schema(model_key: str) -> Dict[str, Any]:
        return RegressionCatalogue.get_hyperparameter_schema(model_key)

    @staticmethod
    def create_regressor(model_key: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return RegressionCatalogue.create_regressor(model_key, params)

    @staticmethod
    def extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return RegressionCatalogue.extract_feature_importances(model, feature_names)
