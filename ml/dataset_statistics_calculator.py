"""
MLForge ML Engine - Extended Dataset Statistics Calculator Module
Provides comprehensive numerical, categorical, quantile matrix, covariance matrix,
bivariate correlation, and distribution fitting calculations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from utils.helpers import make_json_serializable


class ExtendedDatasetStatisticsCalculator:
    """
    Extended Dataset Statistics & Distribution Analysis Suite.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.rows = len(df)
        self.cols = len(df.columns)

    def calculate_full_numerical_matrix(self) -> Dict[str, Any]:
        """
        Calculates numerical statistics matrix across all numeric features.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        matrix = {}

        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty:
                continue

            matrix[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
                "var": round(float(s.var()), 4) if len(s) > 1 else 0.0,
                "median": round(float(s.median()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "q25": round(float(s.quantile(0.25)), 4),
                "q75": round(float(s.quantile(0.75)), 4),
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                "skew": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
                "kurt": round(float(s.kurtosis()), 4) if len(s) > 3 else 0.0
            }

        return matrix

    def calculate_full_categorical_matrix(self) -> Dict[str, Any]:
        """
        Calculates categorical statistics matrix across all text/categorical features.
        """
        cat_df = self.df.select_dtypes(include=['object', 'category', 'bool'])
        matrix = {}

        for col in cat_df.columns:
            s = cat_df[col].dropna()
            if s.empty:
                continue

            counts = s.value_counts()
            top = str(counts.index[0]) if not counts.empty else ""
            freq = int(counts.iloc[0]) if not counts.empty else 0

            matrix[col] = {
                "unique_count": int(s.nunique()),
                "top_value": top,
                "top_frequency": freq,
                "top_ratio": round((freq / max(len(s), 1)) * 100, 2),
                "value_distribution": make_json_serializable(counts.head(5).to_dict())
            }

        return matrix
