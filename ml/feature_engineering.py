"""
MLForge ML Engine - Feature Engineering Module
Performs polynomial feature generation, mathematical log/sqrt transformations,
interaction feature generation, and column selection/dropping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer:
    """
    Feature engineering and mathematical transformation module.
    """
    
    @staticmethod
    def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies log1p transformation to specified numerical columns.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col]):
                # Ensure non-negative values for log
                min_val = df_transformed[col].min()
                shift = abs(min_val) + 1.0 if min_val < 0 else 0.0
                df_transformed[f"{col}_log"] = np.log1p(df_transformed[col] + shift)
        return df_transformed

    @staticmethod
    def apply_sqrt_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies square-root transformation to specified numerical columns.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col]):
                min_val = df_transformed[col].min()
                shift = abs(min_val) if min_val < 0 else 0.0
                df_transformed[f"{col}_sqrt"] = np.sqrt(df_transformed[col] + shift)
        return df_transformed

    @staticmethod
    def apply_interaction_features(df: pd.DataFrame, max_features: int = 5) -> pd.DataFrame:
        """
        Generates pairwise interaction product features between top numerical columns.
        """
        df_transformed = df.copy()
        num_cols = list(df_transformed.select_dtypes(include=[np.number]).columns)[:max_features]
        
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                interaction_name = f"{col1}_x_{col2}"
                df_transformed[interaction_name] = df_transformed[col1] * df_transformed[col2]
                
        return df_transformed

    @staticmethod
    def apply_polynomial_features(
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Generates polynomial features up to specified degree.
        """
        df_transformed = df.copy()
        num_df = df_transformed[columns].fillna(0)
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_arr = poly.fit_transform(num_df)
        poly_cols = poly.get_feature_names_out(columns)
        
        # Add new polynomial columns
        for idx, col_name in enumerate(poly_cols):
            if col_name not in columns:
                df_transformed[col_name] = poly_arr[:, idx]
                
        return df_transformed

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
        """
        Safely drops requested columns from DataFrame.
        """
        existing = [col for col in columns_to_drop if col in df.columns]
        if existing:
            return df.drop(columns=existing)
        return df.copy()
