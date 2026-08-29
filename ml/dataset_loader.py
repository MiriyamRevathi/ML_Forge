"""
MLForge ML Engine - Dataset Loader & Schema Inspection Module
Provides high-performance dataset loading, schema inferencing, memory allocation analysis,
and column classification (numerical vs categorical).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


class DatasetLoader:
    """
    Core engine component for inspecting and loading pandas DataFrames from disk.
    """
    
    @staticmethod
    def load_csv(filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Loads CSV file safely into a Pandas DataFrame.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset CSV file '{filepath}' not found.")
            
        try:
            df = pd.read_csv(filepath, nrows=max_rows)
            if df.empty:
                raise ValueError("Loaded dataset DataFrame is empty.")
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse CSV dataset file: {str(e)}")

    @staticmethod
    def inspect_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts detailed schema metadata from a DataFrame.
        """
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        column_details = []
        for col in df.columns:
            series = df[col]
            column_details.append({
                "name": col,
                "dtype": str(series.dtype),
                "is_numerical": col in num_cols,
                "is_categorical": col in cat_cols,
                "null_count": int(series.isna().sum()),
                "null_percentage": round(float(series.isna().mean() * 100), 2),
                "unique_values_count": int(series.nunique()),
                "sample_values": series.dropna().head(3).tolist()
            })
            
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())
        memory_formatted = f"{memory_usage_bytes / 1024:.2f} KB" if memory_usage_bytes < 1024 * 1024 else f"{memory_usage_bytes / (1024 * 1024):.2f} MB"

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "total_missing_cells": int(df.isna().sum().sum()),
            "total_duplicate_rows": int(df.duplicated().sum()),
            "memory_usage": memory_formatted,
            "columns": column_details
        }

# Feature sync: feature/dataset-management-system (PR #1)
