"""
MLForge ML Engine - Feature Engineering & Transformations Module
Provides mathematical log/sqrt/power transforms, ratio features, interaction products,
date/time component extractors, grouped aggregations, polynomial expansion, and feature selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures


class FeatureTransformationEngine:
    """
    Feature engineering and mathematical transformation suite.
    """

    @staticmethod
    def apply_log_transform(
        df: pd.DataFrame,
        columns: List[str],
        prefix: str = "log_"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies natural log transform (np.log1p) to numerical features.
        """
        df_trans = df.copy()
        new_cols = []

        for col in columns:
            if col in df_trans.columns and pd.api.types.is_numeric_dtype(df_trans[col]):
                new_col_name = f"{prefix}{col}"
                # Shift if negative values present
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
        """
        Applies square-root transform (np.sqrt) to positive numerical features.
        """
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
        """
        Creates ratio features (num / (denom + 1e-6)) between feature pairs.
        """
        df_trans = df.copy()
        new_cols = []

        for num_col, denom_col in feature_pairs:
            if num_col in df_trans.columns and denom_col in df_trans.columns:
                new_col_name = f"{num_col}_div_{denom_col}{suffix}"
                df_trans[new_col_name] = df_trans[num_col] / (df_trans[denom_col].replace(0, 1e-6) + 1e-6)
                new_cols.append(new_col_name)

        return df_trans, new_cols

    @staticmethod
    def create_interaction_products(
        df: pd.DataFrame,
        columns: List[str],
        max_combos: int = 15
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Creates pairwise multiplication features (feat_A * feat_B).
        """
        df_trans = df.copy()
        new_cols = []
        count = 0

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                if col1 in df_trans.columns and col2 in df_trans.columns:
                    new_name = f"{col1}_x_{col2}"
                    df_trans[new_name] = df_trans[col1] * df_trans[col2]
                    new_cols.append(new_name)
                    count += 1
                    if count >= max_combos:
                        break
            if count >= max_combos:
                break

        return df_trans, new_cols

    @staticmethod
    def extract_datetime_components(
        df: pd.DataFrame,
        datetime_columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts year, month, day, dayofweek, is_weekend components from timestamp features.
        """
        df_trans = df.copy()
        new_cols = []

        for col in datetime_columns:
            if col not in df_trans.columns:
                continue
            dt_series = pd.to_datetime(df_trans[col], errors="coerce")
            
            y_col = f"{col}_year"
            m_col = f"{col}_month"
            d_col = f"{col}_day"
            dow_col = f"{col}_dayofweek"
            wk_col = f"{col}_is_weekend"

            df_trans[y_col] = dt_series.dt.year.fillna(-1).astype(int)
            df_trans[m_col] = dt_series.dt.month.fillna(-1).astype(int)
            df_trans[d_col] = dt_series.dt.day.fillna(-1).astype(int)
            df_trans[dow_col] = dt_series.dt.dayofweek.fillna(-1).astype(int)
            df_trans[wk_col] = (dt_series.dt.dayofweek >= 5).astype(int)

            new_cols.extend([y_col, m_col, d_col, dow_col, wk_col])

        return df_trans, new_cols

    @staticmethod
    def select_top_k_features(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_type: str = "classification",
        k: int = 10
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Selects Top K features using ANOVA F-value test (SelectKBest).
        """
        k = min(k, X.shape[1])
        score_func = f_classif if task_type == "classification" else f_regression

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        scores = selector.scores_
        scores = np.nan_to_num(scores, nan=0.0)

        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]

        feature_scores = {feature_names[i]: round(float(scores[i]), 4) for i in range(len(feature_names))}

        return X_selected, selected_names, feature_scores
