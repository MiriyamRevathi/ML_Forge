"""
MLForge ML Engine - Dataset Version Comparison Suite Module
Compares two dataset versions or DataFrames: column diffs, row count deltas,
null count shifts, data type changes, and statistical summary shifts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.dataset_comparison import DatasetComparator


class DatasetComparisonSuite:
    """
    Utility for auditing differences between baseline and target dataset versions suite.
    """

    @staticmethod
    def compare_dataframes(base_df: pd.DataFrame, target_df: pd.DataFrame, base_name: str = "Baseline", target_name: str = "Target") -> Dict[str, Any]:
        return DatasetComparator.compare_dataframes(base_df, target_df, base_name, target_name)
