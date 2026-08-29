"""
MLForge ML Engine - Comprehensive Dataset Profiler Module
Computes row counts, memory footprints, column data type distributions,
duplicate ratios, missingness density, sparsity indices, skewness, kurtosis,
and memory optimization recommendations for Pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DatasetProfiler:
    """
    In-depth dataset profiling and statistical memory auditor.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.total_rows = len(df)
        self.total_cols = len(df.columns)

    def profile_memory_footprint(self) -> Dict[str, Any]:
        """
        Calculates memory usage breakdown by column and offers reduction recommendations.
        """
        memory_usage_series = self.df.memory_usage(deep=True)
        total_bytes = int(memory_usage_series.sum())
        
        col_memory = {}
        recommendations = []
        
        for col in self.df.columns:
            bytes_used = int(memory_usage_series[col])
            dtype = str(self.df[col].dtype)
            col_memory[col] = {
                "bytes": bytes_used,
                "formatted": self._format_bytes(bytes_used),
                "dtype": dtype
            }
            
            # Optimization check for float64 to float32 or object to category
            if dtype == "object":
                num_unique = self.df[col].nunique()
                if self.total_rows > 0 and (num_unique / self.total_rows) < 0.2:
                    recommendations.append(
                        f"Column '{col}' has low unique ratio ({num_unique}/{self.total_rows}). Convert to 'category' dtype to save memory."
                    )
            elif dtype == "int64":
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if c_min >= 0 and c_max < 255:
                    recommendations.append(f"Column '{col}' integer range [{c_min}, {c_max}] can downcast to 'uint8'.")
                elif c_min >= -128 and c_max <= 127:
                    recommendations.append(f"Column '{col}' integer range [{c_min}, {c_max}] can downcast to 'int8'.")

        return {
            "total_bytes": total_bytes,
            "formatted_total": self._format_bytes(total_bytes),
            "column_memory_breakdown": col_memory,
            "optimization_recommendations": recommendations
        }

    def profile_column_types(self) -> Dict[str, Any]:
        """
        Groups columns by inferenced high-level type: numerical, categorical, datetime, boolean, text.
        """
        numerical_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
        boolean_cols = list(self.df.select_dtypes(include=['bool']).columns)
        datetime_cols = list(self.df.select_dtypes(include=['datetime', 'datetime64']).columns)
        
        # High cardinality text vs categorical
        text_cols = []
        clean_cat_cols = []
        for col in categorical_cols:
            if self.df[col].nunique() > 100 and (self.df[col].nunique() / max(self.total_rows, 1)) > 0.4:
                text_cols.append(col)
            else:
                clean_cat_cols.append(col)

        return {
            "total_columns": self.total_cols,
            "numerical_count": len(numerical_cols),
            "categorical_count": len(clean_cat_cols),
            "text_count": len(text_cols),
            "boolean_count": len(boolean_cols),
            "datetime_count": len(datetime_cols),
            "numerical_columns": numerical_cols,
            "categorical_columns": clean_cat_cols,
            "text_columns": text_cols,
            "boolean_columns": boolean_cols,
            "datetime_columns": datetime_cols
        }

    def profile_missingness_and_sparsity(self) -> Dict[str, Any]:
        """
        Measures total missing cells, column missing ratios, and overall dataset matrix sparsity.
        """
        null_matrix = self.df.isna()
        total_cells = max(self.total_rows * self.total_cols, 1)
        total_missing = int(null_matrix.sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2)
        
        col_missingness = {}
        high_missing_cols = []
        
        for col in self.df.columns:
            m_count = int(null_matrix[col].sum())
            m_ratio = round((m_count / max(self.total_rows, 1)) * 100, 2)
            col_missingness[col] = {
                "missing_count": m_count,
                "missing_percentage": m_ratio
            }
            if m_ratio > 40.0:
                high_missing_cols.append(col)
                
        # Calculate zero value sparsity for numerical columns
        num_df = self.df.select_dtypes(include=[np.number])
        zero_count = int((num_df == 0).sum().sum()) if not num_df.empty else 0
        sparsity_pct = round(((total_missing + zero_count) / total_cells) * 100, 2)

        return {
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "overall_missing_percentage": missing_pct,
            "sparsity_percentage": sparsity_pct,
            "columns_missingness": col_missingness,
            "high_missingness_columns": high_missing_cols
        }

    def profile_statistical_distributions(self) -> Dict[str, Any]:
        """
        Calculates skewness, kurtosis, variance, and entropy for columns.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        num_stats = {}
        
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) < 3:
                continue
                
            num_stats[col] = {
                "mean": float(round(series.mean(), 4)),
                "std": float(round(series.std(), 4)),
                "min": float(round(series.min(), 4)),
                "max": float(round(series.max(), 4)),
                "skewness": float(round(series.skew(), 4)),
                "kurtosis": float(round(series.kurtosis(), 4)),
                "is_skewed": abs(float(series.skew())) > 1.0
            }

        cat_df = self.df.select_dtypes(include=['object', 'category'])
        cat_stats = {}
        for col in cat_df.columns:
            series = cat_df[col].dropna()
            if series.empty:
                continue
            probs = series.value_counts(normalize=True)
            entropy = -float((probs * np.log2(probs + 1e-9)).sum())
            cat_stats[col] = {
                "unique_count": int(series.nunique()),
                "top_category": str(series.mode().iloc[0]) if not series.empty else "",
                "top_frequency": int(series.value_counts().iloc[0]) if not series.empty else 0,
                "entropy": round(entropy, 4)
            }

        return {
            "numerical_distribution_stats": num_stats,
            "categorical_distribution_stats": cat_stats
        }

    def generate_complete_profile(self) -> Dict[str, Any]:
        """
        Generates comprehensive profiler summary payload.
        """
        return {
            "dataset_name": self.dataset_name,
            "total_rows": self.total_rows,
            "total_columns": self.total_cols,
            "memory_profile": self.profile_memory_footprint(),
            "type_profile": self.profile_column_types(),
            "missingness_profile": self.profile_missingness_and_sparsity(),
            "distribution_profile": self.profile_statistical_distributions()
        }

    def _format_bytes(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
