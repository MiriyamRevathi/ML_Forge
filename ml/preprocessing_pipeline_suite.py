"""
MLForge ML Engine - Preprocessing Pipeline Suite Module
Constructs scikit-learn ColumnTransformer objects incorporating StandardScaler,
MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, SimpleImputer, and PolynomialFeatures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer


class PreprocessingPipelineSuite:
    """
    Configurable Scikit-Learn Preprocessing Pipeline Assembler.
    """

    @staticmethod
    def build_column_transformer(
        numerical_features: List[str],
        categorical_features: List[str],
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        impute_strategy_num: str = "mean",
        impute_strategy_cat: str = "most_frequent",
        polynomial_degree: Optional[int] = None
    ) -> ColumnTransformer:
        """
        Assembles scikit-learn ColumnTransformer for numerical and categorical features.
        """
        # Numerical Pipeline
        num_steps = [("imputer", SimpleImputer(strategy=impute_strategy_num))]

        if scaling_method == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaling_method == "robust":
            num_steps.append(("scaler", RobustScaler()))

        if polynomial_degree and polynomial_degree > 1:
            num_steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))

        num_pipeline = Pipeline(steps=num_steps)

        # Categorical Pipeline
        cat_steps = [("imputer", SimpleImputer(strategy=impute_strategy_cat, fill_value="Missing"))]

        if encoding_method == "onehot":
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif encoding_method == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        cat_pipeline = Pipeline(steps=cat_steps)

        transformers = []
        if numerical_features:
            transformers.append(("num", num_pipeline, numerical_features))
        if categorical_features:
            transformers.append(("cat", cat_pipeline, categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")
