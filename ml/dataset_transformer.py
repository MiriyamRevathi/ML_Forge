"""
MLForge ML Engine - Dataset Transformer & Manipulation Module
Provides row condition filtering, data type coercion, column renaming,
random/stratified/systematic sampling, dataset merging, and shuffling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union


class DatasetTransformer:
    """
    Data manipulation, sampling, type coercion, and dataset joining utility.
    """

    @staticmethod
    def filter_by_condition(
        df: pd.DataFrame,
        column_name: str,
        operator: str,
        value: Any
    ) -> pd.DataFrame:
        """
        Filters DataFrame rows based on a column condition operator (==, !=, >, <, >=, <=, in, contains).
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")

        series = df[column_name]
        op = operator.lower().strip()

        if op in ["==", "eq", "equal"]:
            mask = series == value
        elif op in ["!=", "ne", "not_equal"]:
            mask = series != value
        elif op in [">", "gt", "greater_than"]:
            mask = series > float(value)
        elif op in ["<", "lt", "less_than"]:
            mask = series < float(value)
        elif op in [">=", "gte", "greater_equal"]:
            mask = series >= float(value)
        elif op in ["<=", "lte", "less_equal"]:
            mask = series <= float(value)
        elif op in ["in", "isin"]:
            val_list = value if isinstance(value, list) else [v.strip() for v in str(value).split(",")]
            mask = series.isin(val_list)
        elif op in ["contains", "str_contains"]:
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        else:
            raise ValueError(f"Unsupported filter operator '{operator}'.")

        return df[mask].copy()

    @staticmethod
    def coerce_column_types(
        df: pd.DataFrame,
        type_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Coerces columns to requested data types ('numeric', 'categorical', 'datetime', 'boolean', 'string').
        """
        df_coerced = df.copy()
        applied_types = {}

        for col, target_type in type_mapping.items():
            if col not in df_coerced.columns:
                continue

            tt = target_type.lower().strip()
            try:
                if tt in ["numeric", "float", "number", "int"]:
                    df_coerced[col] = pd.to_numeric(df_coerced[col], errors="coerce")
                    applied_types[col] = str(df_coerced[col].dtype)
                elif tt in ["category", "categorical"]:
                    df_coerced[col] = df_coerced[col].astype("category")
                    applied_types[col] = "category"
                elif tt in ["str", "string", "text"]:
                    df_coerced[col] = df_coerced[col].astype(str)
                    applied_types[col] = "string"
                elif tt in ["bool", "boolean"]:
                    df_coerced[col] = df_coerced[col].astype(bool)
                    applied_types[col] = "bool"
                elif tt in ["datetime", "date", "timestamp"]:
                    df_coerced[col] = pd.to_datetime(df_coerced[col], errors="coerce")
                    applied_types[col] = "datetime64[ns]"
            except Exception as e:
                applied_types[col] = f"FAILED: {str(e)}"

        return df_coerced, applied_types

    @staticmethod
    def sample_dataset(
        df: pd.DataFrame,
        method: str = "random",
        fraction: float = 0.5,
        n_samples: Optional[int] = None,
        stratify_column: Optional[str] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Samples dataset using Random Sampling, Stratified Sampling, or Systematic Sampling.
        """
        m = method.lower().strip()
        total_rows = len(df)

        if total_rows == 0:
            return df.copy()

        if n_samples is None:
            n_samples = max(int(total_rows * fraction), 1)
        n_samples = min(n_samples, total_rows)

        if m == "stratified" and stratify_column and stratify_column in df.columns:
            try:
                groups = df.groupby(stratify_column, group_keys=False)
                sampled = groups.apply(lambda x: x.sample(frac=n_samples / total_rows, random_state=random_state))
                return sampled.copy()
            except Exception:
                # Fallback to random sample
                return df.sample(n=n_samples, random_state=random_state).copy()

        elif m == "systematic":
            step = max(int(total_rows / n_samples), 1)
            indices = list(range(0, total_rows, step))[:n_samples]
            return df.iloc[indices].copy()

        else:
            # Default Random Sampling
            return df.sample(n=n_samples, random_state=random_state).copy()

    @staticmethod
    def rename_columns(
        df: pd.DataFrame,
        rename_map: Optional[Dict[str, str]] = None,
        strip_whitespace: bool = True,
        lowercase: bool = False,
        replace_spaces_with_underscore: bool = True
    ) -> pd.DataFrame:
        """
        Renames and cleans column headers.
        """
        df_clean = df.copy()
        
        if strip_whitespace:
            df_clean.columns = [col.strip() for col in df_clean.columns]
        if replace_spaces_with_underscore:
            df_clean.columns = [col.replace(" ", "_") for col in df_clean.columns]
        if lowercase:
            df_clean.columns = [col.lower() for col in df_clean.columns]
            
        if rename_map:
            df_clean = df_clean.rename(columns=rename_map)

        return df_clean

    @staticmethod
    def merge_datasets(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on_column: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "inner"
    ) -> pd.DataFrame:
        """
        Merges two DataFrames on key columns (inner, left, right, outer).
        """
        return pd.merge(
            left_df,
            right_df,
            on=on_column,
            left_on=left_on,
            right_on=right_on,
            how=how
        )

    @staticmethod
    def concatenate_datasets(
        df_list: List[pd.DataFrame],
        axis: int = 0,
        ignore_index: bool = True
    ) -> pd.DataFrame:
        """
        Concatenates a list of DataFrames along rows (axis=0) or columns (axis=1).
        """
        return pd.concat(df_list, axis=axis, ignore_index=ignore_index)
