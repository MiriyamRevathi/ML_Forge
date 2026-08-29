"""
MLForge ML Engine - Extended Classification Catalogue Suite Module
Provides scikit-learn wrappers, hyperparameter schemas, fitting methods,
class probability calculations, and feature importance extractors for 10 classification algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from ml.models.classification import ClassificationCatalogue


class ExtendedClassificationCatalogueSuite:
    """
    Catalogue and factory for 10 production Classification algorithms.
    """

    SUPPORTED_CLASSIFIERS = ClassificationCatalogue.SUPPORTED_CLASSIFIERS

    @staticmethod
    def get_hyperparameter_schema(model_key: str) -> Dict[str, Any]:
        return ClassificationCatalogue.get_hyperparameter_schema(model_key)

    @staticmethod
    def create_classifier(model_key: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return ClassificationCatalogue.create_classifier(model_key, params)

    @staticmethod
    def extract_feature_importances(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        return ClassificationCatalogue.extract_feature_importances(model, feature_names)
