"""
MLForge ML Engine - Data Cleaning Module
Handles missing value imputation (Mean, Median, Mode, Constant, Drop),
duplicate row removal, and outlier detection/filtering (IQR & Z-score).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DataCleaner:
    """
    Data cleaning and outlier handling transformer.
    """

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Removes duplicate rows from DataFrame. Returns cleaned DataFrame and count of dropped rows.
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates().copy()
        dropped_count = initial_rows - len(df_clean)
        return df_clean, dropped_count

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Imputes or drops missing values based on strategy ('mean', 'median', 'mode', 'constant', 'drop').
        """
        df_clean = df.copy()
        initial_missing = int(df_clean.isna().sum().sum())
        
        if initial_missing == 0:
            return df_clean, {"initial_missing": 0, "remaining_missing": 0, "strategy": strategy}

        # If target_column has missing values, drop those rows first
        if target_column and target_column in df_clean.columns:
            df_clean = df_clean.dropna(subset=[target_column])

        if strategy == "drop":
            df_clean = df_clean.dropna().copy()
        else:
            num_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
            cat_cols = list(df_clean.select_dtypes(include=['object', 'category', 'bool']).columns)
            
            # Impute numerical columns
            for col in num_cols:
                if df_clean[col].isna().sum() > 0:
                    if strategy == "mean":
                        val = df_clean[col].mean()
                    elif strategy == "median":
                        val = df_clean[col].median()
                    elif strategy == "mode":
                        val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 0
                    elif strategy == "constant":
                        val = fill_value if fill_value is not None else 0
                    else:
                        val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(val)

            # Impute categorical columns with most frequent / mode
            for col in cat_cols:
                if df_clean[col].isna().sum() > 0:
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Missing"
                    df_clean[col] = df_clean[col].fillna(mode_val)

        remaining_missing = int(df_clean.isna().sum().sum())
        return df_clean, {
            "initial_missing": initial_missing,
            "remaining_missing": remaining_missing,
            "strategy": strategy
        }

    @staticmethod
    def remove_outliers_iqr(
        df: pd.DataFrame,
        factor: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Filters outliers using Interquartile Range (IQR) method.
        """
        df_clean = df.copy()
        num_cols = columns or list(df_clean.select_dtypes(include=[np.number]).columns)
        initial_rows = len(df_clean)
        
        mask = pd.Series(True, index=df_clean.index)
        for col in num_cols:
            if col in df_clean.columns:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    mask = mask & (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    
        df_filtered = df_clean[mask].copy()
        removed_count = initial_rows - len(df_filtered)
        return df_filtered, removed_count

    @staticmethod
    def remove_outliers_zscore(
        df: pd.DataFrame,
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Filters outliers using Z-Score method (|Z| > threshold).
        """
        df_clean = df.copy()
        num_cols = columns or list(df_clean.select_dtypes(include=[np.number]).columns)
        initial_rows = len(df_clean)
        
        mask = pd.Series(True, index=df_clean.index)
        for col in num_cols:
            if col in df_clean.columns:
                std = df_clean[col].std()
                if std > 0:
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / std)
                    mask = mask & (z_scores <= threshold)
                    
        df_filtered = df_clean[mask].copy()
        removed_count = initial_rows - len(df_filtered)
        return df_filtered, removed_count

# Feature sync: feature/data-cleaning-normalizer (PR #4)
