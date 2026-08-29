"""
MLForge ML Engine - Dataset Version Comparison Module
Compares two dataset versions or DataFrames: column diffs, row count deltas,
null count shifts, data type changes, and statistical summary shifts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DatasetComparator:
    """
    Utility for auditing differences between baseline and target dataset versions.
    """

    @staticmethod
    def compare_dataframes(
        base_df: pd.DataFrame,
        target_df: pd.DataFrame,
        base_name: str = "Baseline",
        target_name: str = "Target"
    ) -> Dict[str, Any]:
        """
        Calculates complete diff analysis between baseline and target DataFrames.
        """
        base_cols = set(base_df.columns)
        target_cols = set(target_df.columns)

        added_cols = list(target_cols - base_cols)
        removed_cols = list(base_cols - target_cols)
        common_cols = list(base_cols.intersection(target_cols))

        row_delta = len(target_df) - len(base_df)
        row_delta_pct = round((row_delta / max(len(base_df), 1)) * 100, 2)

        # Data type changes on common columns
        dtype_changes = []
        null_changes = []
        summary_shifts = []

        for col in common_cols:
            b_type = str(base_df[col].dtype)
            t_type = str(target_df[col].dtype)
            if b_type != t_type:
                dtype_changes.append({
                    "column": col,
                    "baseline_type": b_type,
                    "target_type": t_type
                })

            b_nulls = int(base_df[col].isna().sum())
            t_nulls = int(target_df[col].isna().sum())
            if b_nulls != t_nulls:
                null_changes.append({
                    "column": col,
                    "baseline_nulls": b_nulls,
                    "target_nulls": t_nulls,
                    "null_delta": t_nulls - b_nulls
                })

            # Statistical shift for numerical columns
            if pd.api.types.is_numeric_dtype(base_df[col]) and pd.api.types.is_numeric_dtype(target_df[col]):
                b_mean = float(base_df[col].mean()) if not base_df[col].dropna().empty else 0.0
                t_mean = float(target_df[col].mean()) if not target_df[col].dropna().empty else 0.0
                mean_diff = t_mean - b_mean
                mean_diff_pct = round((mean_diff / (abs(b_mean) if b_mean != 0 else 1.0)) * 100, 2)

                summary_shifts.append({
                    "column": col,
                    "baseline_mean": round(b_mean, 4),
                    "target_mean": round(t_mean, 4),
                    "mean_delta": round(mean_diff, 4),
                    "mean_delta_percentage": mean_diff_pct
                })

        return {
            "baseline_name": base_name,
            "target_name": target_name,
            "baseline_shape": [len(base_df), len(base_df.columns)],
            "target_shape": [len(target_df), len(target_df.columns)],
            "row_count_delta": row_delta,
            "row_count_delta_percentage": row_delta_pct,
            "added_columns": added_cols,
            "removed_columns": removed_cols,
            "common_columns_count": len(common_cols),
            "data_type_changes": dtype_changes,
            "null_count_changes": null_changes,
            "numerical_summary_shifts": summary_shifts
        }
