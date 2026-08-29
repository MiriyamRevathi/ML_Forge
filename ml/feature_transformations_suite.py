"""
MLForge ML Engine - Extended Feature Engineering & Transformations Suite Module
Provides mathematical log, square root, power transforms, ratio features, interaction products,
date/time component extractors, grouped aggregations, polynomial expansion, and feature selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression


class FeatureTransformationsSuite:
    """
    Feature Engineering & Feature Selection Suite.
    """

    @staticmethod
    def apply_log_transform(
        df: pd.DataFrame,
        columns: List[str],
        prefix: str = "log_"
    ) -> Tuple[pd.DataFrame, List[str]]:
        df_trans = df.copy()
        new_cols = []

        for col in columns:
            if col in df_trans.columns and pd.api.types.is_numeric_dtype(df_trans[col]):
                new_col_name = f"{prefix}{col}"
                min_val = df_trans[col].min()
                shift = abs(min_val) + 1.0 if min_val < 0 else 0.0
                df_trans[new_col_name] = np.log1p(df_trans[col] + shift)
                new_cols.append(new_col_name)

        return df_trans, new_cols

    @staticmethod
    def apply_sqrt_transform(
        df: pd.DataFrame,
        columns: List[str],
        prefix: str = "sqrt_"
    ) -> Tuple[pd.DataFrame, List[str]]:
        df_trans = df.copy()
        new_cols = []

        for col in columns:
            if col in df_trans.columns and pd.api.types.is_numeric_dtype(df_trans[col]):
                new_col_name = f"{prefix}{col}"
                min_val = df_trans[col].min()
                shift = abs(min_val) if min_val < 0 else 0.0
                df_trans[new_col_name] = np.sqrt(df_trans[col] + shift)
                new_cols.append(new_col_name)

        return df_trans, new_cols

    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]],
        suffix: str = "_ratio"
    ) -> Tuple[pd.DataFrame, List[str]]:
        df_trans = df.copy()
        new_cols = []

        for num_col, denom_col in feature_pairs:
            if num_col in df_trans.columns and denom_col in df_trans.columns:
                new_col_name = f"{num_col}_div_{denom_col}{suffix}"
                df_trans[new_col_name] = df_trans[num_col] / (df_trans[denom_col].replace(0, 1e-6) + 1e-6)
                new_cols.append(new_col_name)

        return df_trans, new_cols

    @staticmethod
    def filter_low_variance_features(
        df: pd.DataFrame,
        threshold: float = 0.0
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Removes constant or low-variance numerical features (VarianceThreshold).
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return df.copy(), []

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(num_df)

        retained_indices = selector.get_support(indices=True)
        retained_cols = [num_df.columns[i] for i in retained_indices]
        dropped_cols = [col for col in num_df.columns if col not in retained_cols]

        df_filtered = df.drop(columns=dropped_cols)
        return df_filtered, dropped_cols
