"""
MLForge ML Engine - Extended Dataset Statistics Engine Module
Computes parametric and non-parametric summary statistics, quantile matrices,
covariance matrices, rank correlations, Chi-Square test of independence,
ANOVA F-tests, and statistical distribution fitting.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from utils.helpers import make_json_serializable


class DatasetStatisticsEngine:
    """
    Extended Statistical Computation & Hypothesis Testing Suite.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.total_rows = len(df)
        self.total_cols = len(df.columns)

    def compute_extended_numerical_stats(self) -> Dict[str, Any]:
        """
        Calculates extended numerical statistics: mean, std, variance, median, IQR,
        quantiles (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%), skewness, kurtosis,
        standard error of mean, and trimean.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        results = {}

        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) < 3:
                continue

            q01 = float(series.quantile(0.01))
            q05 = float(series.quantile(0.05))
            q10 = float(series.quantile(0.10))
            q25 = float(series.quantile(0.25))
            q50 = float(series.median())
            q75 = float(series.quantile(0.75))
            q90 = float(series.quantile(0.90))
            q95 = float(series.quantile(0.95))
            q99 = float(series.quantile(0.99))

            iqr = q75 - q25
            trimean = (q25 + 2 * q50 + q75) / 4.0
            sem = float(series.sem()) if len(series) > 1 else 0.0

            results[col] = {
                "count": len(series),
                "null_count": int(num_df[col].isna().sum()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "sem": round(sem, 4),
                "variance": round(float(series.var()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "range": round(float(series.max() - series.min()), 4),
                "trimean": round(trimean, 4),
                "quantiles": {
                    "p01": round(q01, 4),
                    "p05": round(q05, 4),
                    "p10": round(q10, 4),
                    "p25": round(q25, 4),
                    "p50": round(q50, 4),
                    "p75": round(q75, 4),
                    "p90": round(q90, 4),
                    "p95": round(q95, 4),
                    "p99": round(q99, 4)
                },
                "iqr": round(iqr, 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4)
            }

        return results

    def compute_categorical_contingency(self, col1: str, col2: str) -> Dict[str, Any]:
        """
        Performs Chi-Square test of independence between two categorical features.
        """
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise KeyError(f"Columns '{col1}' or '{col2}' not found.")

        contingency_matrix = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_matrix)

        return {
            "feature_1": col1,
            "feature_2": col2,
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_val), 6),
            "degrees_of_freedom": int(dof),
            "is_statistically_dependent": bool(p_val < 0.05),
            "contingency_matrix": contingency_matrix.to_dict()
        }

    def compute_anova_ftest(self, categorical_col: str, numerical_col: str) -> Dict[str, Any]:
        """
        Performs One-Way ANOVA F-test comparing numerical means across categorical groups.
        """
        if categorical_col not in self.df.columns or numerical_col not in self.df.columns:
            raise KeyError(f"Columns '{categorical_col}' or '{numerical_col}' not found.")

        groups = [group[numerical_col].dropna().values for _, group in self.df.groupby(categorical_col)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return {"f_statistic": 0.0, "p_value": 1.0, "is_significant": False}

        f_stat, p_val = stats.f_oneway(*groups)

        return {
            "categorical_feature": categorical_col,
            "numerical_feature": numerical_col,
            "f_statistic": round(float(f_stat), 4) if not np.isnan(f_stat) else 0.0,
            "p_value": round(float(p_val), 6) if not np.isnan(p_val) else 1.0,
            "is_statistically_significant": bool(p_val < 0.05) if not np.isnan(p_val) else False
        }
