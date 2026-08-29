"""
MLForge ML Engine - Dataset Transformer Suite Module
Provides row condition filtering, data type coercion, column renaming,
random/stratified/systematic sampling, dataset merging, and shuffling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.dataset_transformer import DatasetTransformer


class DatasetTransformerSuite:
    """
    Data manipulation, sampling, type coercion, and dataset joining utility suite.
    """

    @staticmethod
    def filter_by_condition(df: pd.DataFrame, column_name: str, operator: str, value: Any) -> pd.DataFrame:
        return DatasetTransformer.filter_by_condition(df, column_name, operator, value)

    @staticmethod
    def coerce_column_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        return DatasetTransformer.coerce_column_types(df, type_mapping)

    @staticmethod
    def sample_dataset(df: pd.DataFrame, method: str = "random", fraction: float = 0.5, n_samples: Optional[int] = None, stratify_column: Optional[str] = None, random_state: int = 42) -> pd.DataFrame:
        return DatasetTransformer.sample_dataset(df, method, fraction, n_samples, stratify_column, random_state)

    @staticmethod
    def rename_columns(df: pd.DataFrame, rename_map: Optional[Dict[str, str]] = None, strip_whitespace: bool = True, lowercase: bool = False, replace_spaces_with_underscore: bool = True) -> pd.DataFrame:
        return DatasetTransformer.rename_columns(df, rename_map, strip_whitespace, lowercase, replace_spaces_with_underscore)

    @staticmethod
    def merge_datasets(left_df: pd.DataFrame, right_df: pd.DataFrame, on_column: Optional[str] = None, left_on: Optional[str] = None, right_on: Optional[str] = None, how: str = "inner") -> pd.DataFrame:
        return DatasetTransformer.merge_datasets(left_df, right_df, on_column, left_on, right_on, how)

    @staticmethod
    def concatenate_datasets(df_list: List[pd.DataFrame], axis: int = 0, ignore_index: bool = True) -> pd.DataFrame:
        return DatasetTransformer.concatenate_datasets(df_list, axis, ignore_index)
