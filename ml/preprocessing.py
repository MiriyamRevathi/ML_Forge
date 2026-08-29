"""
MLForge ML Engine - Preprocessing & Feature Scaling Module
Constructs scikit-learn preprocessing transformers (StandardScaler, MinMaxScaler,
RobustScaler, OneHotEncoder, OrdinalEncoder, ColumnTransformer).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class FeaturePreprocessor:
    """
    Configurable scikit-learn feature preprocessor pipeline.
    """
    
    def __init__(
        self,
        scaler_type: str = "standard",
        encoder_type: str = "onehot",
        impute_strategy: str = "mean"
    ):
        self.scaler_type = scaler_type.lower()
        self.encoder_type = encoder_type.lower()
        self.impute_strategy = impute_strategy.lower()
        self.column_transformer: Optional[ColumnTransformer] = None
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_out: List[str] = []

    def _get_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def _get_encoder(self):
        if self.encoder_type == "ordinal":
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit_transform(
        self,
        X: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fits column transformer on training features and transforms X into numerical matrix.
        """
        X = X.copy()
        
        self.numerical_features = numerical_cols if numerical_cols is not None else list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_features = categorical_cols if categorical_cols is not None else list(X.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        transformers = []
        
        if self.numerical_features:
            num_pipe = [
                ("imputer", SimpleImputer(strategy=self.impute_strategy if self.impute_strategy in ["mean", "median"] else "mean")),
                ("scaler", self._get_scaler())
            ]
            from sklearn.pipeline import Pipeline
            transformers.append(("num", Pipeline(num_pipe), self.numerical_features))
            
        if self.categorical_features:
            cat_pipe = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", self._get_encoder())
            ]
            from sklearn.pipeline import Pipeline
            transformers.append(("cat", Pipeline(cat_pipe), self.categorical_features))
            
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        
        X_transformed = self.column_transformer.fit_transform(X)
        
        # Determine feature names out
        feature_names = []
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
            
        if self.categorical_features:
            if self.encoder_type == "onehot":
                try:
                    encoder = self.column_transformer.named_transformers_["cat"].named_steps["encoder"]
                    encoded_names = list(encoder.get_feature_names_out(self.categorical_features))
                    feature_names.extend(encoded_names)
                except Exception:
                    feature_names.extend([f"cat_{i}" for i in range(X_transformed.shape[1] - len(self.numerical_features))])
            else:
                feature_names.extend(self.categorical_features)
                
        self.feature_names_out = feature_names
        return X_transformed, self.feature_names_out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms new incoming features using fitted ColumnTransformer.
        """
        if self.column_transformer is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before calling transform().")
            
        return self.column_transformer.transform(X)

# Feature sync: feature/preprocessing-feature-engineering (PR #5)

# Feature sync: feature/preprocessing-feature-engineering (PR #5)

# Feature sync: feature/preprocessing-feature-engineering (PR #5)
