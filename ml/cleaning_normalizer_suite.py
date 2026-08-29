"""
MLForge ML Engine - Data Cleaning & Normalization Engine Suite Module
Provides configurable strategy objects for missing value imputation, outlier handling,
string normalization, category frequency lumping, type coercion, and column pruning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union


class CleaningNormalizerSuite:
    """
    Data Cleaning & Normalization Engine Suite.
    """

    @staticmethod
    def impute_missing_values(
        df: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Imputes missing values using Mean, Median, Mode, Constant, or Row Dropping.
        """
        df_clean = df.copy()
        target_cols = columns or list(df_clean.columns)
        strat = strategy.lower().strip()

        if strat == "drop":
            return df_clean.dropna(subset=target_cols).copy()

        for col in target_cols:
            if col not in df_clean.columns:
                continue
            if df_clean[col].isna().sum() == 0:
                continue

            if strat == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(val)
            elif strat == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(val)
            elif strat == "mode":
                mode_res = df_clean[col].mode()
                val = mode_res.iloc[0] if not mode_res.empty else ("Missing" if not pd.api.types.is_numeric_dtype(df_clean[col]) else 0)
                df_clean[col] = df_clean[col].fillna(val)
            elif strat in ["constant", "value"]:
                val = fill_value if fill_value is not None else ("Missing" if not pd.api.types.is_numeric_dtype(df_clean[col]) else 0)
                df_clean[col] = df_clean[col].fillna(val)

        return df_clean

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "clip",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detects and handles outliers via IQR bounds or Z-score thresholds.
        """
        df_out = df.copy()
        num_cols = columns or list(df_out.select_dtypes(include=[np.number]).columns)
        m = method.lower().strip()
        act = action.lower().strip()

        for col in num_cols:
            if col not in df_out.columns:
                continue
            series = df_out[col].dropna()
            if series.empty:
                continue

            if m == "iqr":
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - (threshold * iqr)
                upper_bound = q75 + (threshold * iqr)
            elif m == "zscore":
                mean, std = series.mean(), series.std()
                if std == 0:
                    continue
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
            else:
                continue

            if act == "clip":
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif act == "nullify":
                mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
                df_out.loc[mask, col] = np.nan
            elif act == "drop":
                mask = (df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)
                df_out = df_out[mask]

        return df_out.copy()

    @staticmethod
    def normalize_strings(
        df: pd.DataFrame,
        strip_whitespace: bool = True,
        lowercase: bool = True,
        remove_special_chars: bool = False,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalizes string/categorical text columns.
        """
        df_norm = df.copy()
        cat_cols = columns or list(df_norm.select_dtypes(include=['object', 'category']).columns)

        for col in cat_cols:
            if col not in df_norm.columns:
                continue
            s = df_norm[col].astype(str)
            if strip_whitespace:
                s = s.str.strip()
            if lowercase:
                s = s.str.lower()
            if remove_special_chars:
                s = s.str.replace(r'[^a-zA-Z0-9\s_]', '', regex=True)
            df_norm[col] = s

        return df_norm

    @staticmethod
    def group_rare_categories(
        df: pd.DataFrame,
        threshold_percentage: float = 2.0,
        other_label: str = "Other",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Groups infrequent categorical values into 'Other'.
        """
        df_grouped = df.copy()
        cat_cols = columns or list(df_grouped.select_dtypes(include=['object', 'category']).columns)
        total_rows = len(df_grouped)

        if total_rows == 0:
            return df_grouped

        for col in cat_cols:
            if col not in df_grouped.columns:
                continue
            counts = df_grouped[col].value_counts()
            freq_pcts = (counts / total_rows) * 100
            rare_levels = freq_pcts[freq_pcts < threshold_percentage].index.tolist()

            if rare_levels:
                df_grouped[col] = df_grouped[col].apply(lambda val: other_label if val in rare_levels else val)

        return df_grouped
