"""
MLForge ML Engine - Detailed Exploratory Data Analysis (EDA) Statistics Module
Computes numerical descriptive statistics, quantiles, skewness, kurtosis, variance,
categorical frequencies, entropy, and bivariate correlation matrices for DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from utils.helpers import make_json_serializable


class EDAStatisticsCalculator:
    """
    EDA Statistical Metrics Calculation Suite.
    """

    @staticmethod
    def calculate_numerical_summaries(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates complete numerical feature summary stats: count, mean, std, min,
        q25, median, q75, max, IQR, variance, skewness, kurtosis.
        """
        num_df = df.select_dtypes(include=[np.number])
        results = {}

        for col in num_df.columns:
            series = num_df[col].dropna()
            if series.empty:
                continue

            q25 = float(series.quantile(0.25))
            median = float(series.median())
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            results[col] = {
                "count": int(len(series)),
                "missing_count": int(num_df[col].isna().sum()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "variance": round(float(series.var()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "q25": round(q25, 4),
                "median": round(median, 4),
                "q75": round(q75, 4),
                "max": round(float(series.max()), 4),
                "iqr": round(iqr, 4),
                "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                "kurtosis": round(float(series.kurtosis()), 4) if len(series) > 3 else 0.0
            }

        return results

    @staticmethod
    def calculate_categorical_summaries(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates categorical feature frequency metrics: unique count, top value,
        mode frequency, mode percentage, value counts, and Shannon entropy.
        """
        cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
        results = {}

        for col in cat_df.columns:
            series = cat_df[col].dropna()
            if series.empty:
                continue

            total_count = len(series)
            val_counts = series.value_counts()
            top_val = str(series.mode().iloc[0]) if not series.empty else ""
            top_freq = int(val_counts.iloc[0]) if not val_counts.empty else 0
            top_pct = round((top_freq / max(total_count, 1)) * 100, 2)

            probs = val_counts / max(total_count, 1)
            entropy = -float((probs * np.log2(probs + 1e-9)).sum())

            results[col] = {
                "count": total_count,
                "missing_count": int(cat_df[col].isna().sum()),
                "unique_count": int(series.nunique()),
                "top_category": top_val,
                "top_frequency": top_freq,
                "top_percentage": top_pct,
                "entropy": round(entropy, 4),
                "value_counts": make_json_serializable(val_counts.head(10).to_dict())
            }

        return results

    @staticmethod
    def calculate_bivariate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes Pearson and Spearman correlation matrices for numerical columns.
        """
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty or len(num_df.columns) < 2:
            return {"pearson": {"columns": [], "matrix": []}, "spearman": {"columns": [], "matrix": []}}

        pearson_corr = num_df.corr(method="pearson").round(4).fillna(0)
        spearman_corr = num_df.corr(method="spearman").round(4).fillna(0)

        return {
            "pearson": {
                "columns": list(pearson_corr.columns),
                "matrix": pearson_corr.values.tolist()
            },
            "spearman": {
                "columns": list(spearman_corr.columns),
                "matrix": spearman_corr.values.tolist()
            }
        }
