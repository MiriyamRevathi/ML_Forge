"""
MLForge ML Engine - Dataset Profiler Suite Module
Computes row counts, memory footprints, column data type distributions,
duplicate ratios, missingness density, sparsity indices, skewness, kurtosis,
and memory optimization recommendations for Pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.dataset_profiler import DatasetProfiler


class DatasetProfilerSuite:
    """
    In-depth dataset profiling and statistical memory auditor suite.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.profiler = DatasetProfiler(df, dataset_name=dataset_name)

    def profile_memory_footprint(self) -> Dict[str, Any]:
        return self.profiler.profile_memory_footprint()

    def profile_column_types(self) -> Dict[str, Any]:
        return self.profiler.profile_column_types()

    def profile_missingness_and_sparsity(self) -> Dict[str, Any]:
        return self.profiler.profile_missingness_and_sparsity()

    def profile_statistical_distributions(self) -> Dict[str, Any]:
        return self.profiler.profile_statistical_distributions()

    def generate_complete_profile(self) -> Dict[str, Any]:
        return self.profiler.generate_complete_profile()
