"""
MLForge ML Engine - Dataset Validator & Schema Contract Engine Module
Provides validation of column data types, row count bounds, target column requirements,
non-empty constraints, missingness limits, and schema contract enforcement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DatasetSchemaContract:
    """
    Schema Contract Definition for Datasets.
    """

    def __init__(self, expected_columns: Dict[str, str], target_column: Optional[str] = None):
        self.expected_columns = expected_columns
        self.target_column = target_column

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates DataFrame against the expected schema contract.
        """
        errors = []

        if df.empty:
            errors.append("DataFrame is completely empty (0 rows).")
            return False, errors

        # Check required columns
        for col, exp_type in self.expected_columns.items():
            if col not in df.columns:
                errors.append(f"Missing required column '{col}'.")
            else:
                actual_dtype = str(df[col].dtype)
                if exp_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' expected numeric type, got '{actual_dtype}'.")
                elif exp_type == "categorical" and pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' expected categorical/string type, got '{actual_dtype}'.")

        # Check target column
        if self.target_column and self.target_column not in df.columns:
            errors.append(f"Target column '{self.target_column}' is missing from DataFrame.")

        return len(errors) == 0, errors


class DatasetValidatorEngine:
    """
    Dataset Integrity & Schema Validation Suite.
    """

    @staticmethod
    def validate_dataset_for_ml(
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification"
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Performs deep validation on DataFrame suitability for machine learning pipeline.
        """
        errors = []
        info = {}

        if df.empty:
            errors.append("Dataset contains 0 rows.")
            return False, errors, info

        if len(df.columns) < 2:
            errors.append("Dataset must contain at least 1 feature column and 1 target column.")

        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found in dataset columns.")
            return False, errors, info

        target_series = df[target_column].dropna()
        if target_series.empty:
            errors.append(f"Target column '{target_column}' contains only NaN values.")

        if task_type == "classification":
            unique_classes = target_series.nunique()
            if unique_classes < 2:
                errors.append(f"Classification target '{target_column}' must have at least 2 distinct classes (found {unique_classes}).")
            elif unique_classes > 100 and (unique_classes / len(target_series)) > 0.5:
                errors.append(f"Target '{target_column}' has too many unique values ({unique_classes}) for classification. Did you mean regression?")
            info["unique_target_classes"] = unique_classes

        elif task_type == "regression":
            if not pd.api.types.is_numeric_dtype(target_series):
                errors.append(f"Regression target '{target_column}' must be numerical (got {target_series.dtype}).")

        info["total_rows"] = len(df)
        info["total_columns"] = len(df.columns)
        info["feature_columns_count"] = len(df.columns) - 1

        return len(errors) == 0, errors, info
