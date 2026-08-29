"""
MLForge ML Engine - Cleaning & Normalization Facade Module
Provides high-level API facade for dataset cleaning, imputation, outlier handling,
string normalization, type casting, and column pruning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.cleaning_normalizer_suite import CleaningNormalizerSuite


class CleaningNormalizerFacade:
    """
    High-level facade for Data Cleaning & Normalization workflows.
    """

    @staticmethod
    def execute_cleaning_pipeline(
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        missing_strategy: str = "mean",
        outlier_method: str = "none",
        outlier_threshold: float = 1.5,
        normalize_strings: bool = True,
        group_rare_cats: bool = False
    ) -> pd.DataFrame:
        """
        Executes multi-step data cleaning pipeline.
        """
        df_clean = df.copy()

        if drop_duplicates:
            df_clean = df_clean.drop_duplicates().copy()

        if missing_strategy != "none":
            df_clean = CleaningNormalizerSuite.impute_missing_values(df_clean, strategy=missing_strategy)

        if outlier_method != "none":
            df_clean = CleaningNormalizerSuite.handle_outliers(df_clean, method=outlier_method, threshold=outlier_threshold, action="clip")

        if normalize_strings:
            df_clean = CleaningNormalizerSuite.normalize_strings(df_clean, strip_whitespace=True, lowercase=True)

        if group_rare_cats:
            df_clean = CleaningNormalizerSuite.group_rare_categories(df_clean, threshold_percentage=2.0)

        return df_clean
