"""
MLForge ML Engine - Dataset Analyzer & Structural Auditor Module
Provides structural dataset auditing, column missingness pattern detection,
data type classification, row duplicate analysis, and matrix sparsity calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from utils.helpers import make_json_serializable


class DatasetStructuralAnalyzer:
    """
    Dataset Structural Inspection & Analysis Suite.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.rows = len(df)
        self.cols = len(df.columns)

    def analyze_structure(self) -> Dict[str, Any]:
        """
        Calculates complete dataset structural analysis.
        """
        shape_info = {
            "rows_count": self.rows,
            "columns_count": self.cols,
            "total_cells": self.rows * self.cols
        }

        # Data type breakdown
        dtypes_count = self.df.dtypes.value_counts().to_dict()
        formatted_dtypes = {str(k): int(v) for k, v in dtypes_count.items()}

        # Column detail table
        col_details = []
        for col in self.df.columns:
            series = self.df[col]
            null_cnt = int(series.isna().sum())
            null_pct = round((null_cnt / max(self.rows, 1)) * 100, 2)
            unique_cnt = int(series.nunique(dropna=True))

            col_type = "numerical" if pd.api.types.is_numeric_dtype(series) else ("categorical" if pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series) else "datetime")

            col_details.append({
                "column_name": col,
                "data_type": str(series.dtype),
                "inferred_type": col_type,
                "missing_count": null_cnt,
                "missing_percentage": null_pct,
                "unique_values_count": unique_cnt,
                "is_constant": unique_cnt <= 1,
                "sample_values": make_json_serializable(series.dropna().head(3).tolist())
            })

        # Memory usage
        memory_bytes = int(self.df.memory_usage(deep=True).sum())

        return {
            "dataset_name": self.dataset_name,
            "shape": shape_info,
            "data_type_counts": formatted_dtypes,
            "columns": col_details,
            "memory_usage_bytes": memory_bytes,
            "formatted_memory": self._format_size(memory_bytes)
        }

    def _format_size(self, bytes_size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
