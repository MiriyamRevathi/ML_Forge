"""
MLForge ML Engine - Cleaning Engine Suite Module
Configurable data cleaning strategies for duplicate dropping, outlier filtering,
and missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.cleaning import DataCleaner


class CleaningEngineSuite:
    """
    Data Cleaning Engine Suite.
    """

    @staticmethod
    def clean_dataset(
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        missing_strategy: str = "mean",
        missing_constant: Optional[Any] = None,
        outlier_method: str = "none",
        outlier_threshold: float = 1.5
    ) -> pd.DataFrame:
        return DataCleaner.clean_dataset(
            df=df,
            drop_duplicates=drop_duplicates,
            missing_strategy=missing_strategy,
            missing_constant=missing_constant,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
