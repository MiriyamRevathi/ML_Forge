"""
MLForge ML Engine - Preprocessing Engine Suite Module
Scikit-learn ColumnTransformer builder incorporating StandardScaler, MinMaxScaler, RobustScaler,
OneHotEncoder, OrdinalEncoder, and SimpleImputer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ml.preprocessing import PreprocessorBuilder, ColumnTransformer


class PreprocessingEngineSuite:
    """
    Scikit-Learn Preprocessing Pipeline Builder Suite.
    """

    @staticmethod
    def build_preprocessor(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        scaling_method: str = "standard",
        encoding_method: str = "onehot"
    ) -> ColumnTransformer:
        return PreprocessorBuilder.build_preprocessor(
            df=df,
            target_column=target_column,
            scaling_method=scaling_method,
            encoding_method=encoding_method
        )
