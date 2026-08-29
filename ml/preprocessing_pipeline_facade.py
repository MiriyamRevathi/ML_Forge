"""
MLForge ML Engine - Preprocessing Pipeline Facade Module
High-level facade for scikit-learn ColumnTransformer construction, feature inspection,
scaling configuration, and encoding choice.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ml.preprocessing_pipeline_suite import PreprocessingPipelineSuite
from sklearn.compose import ColumnTransformer


class PreprocessingPipelineFacade:
    """
    High-level facade for Scikit-Learn Preprocessing Pipeline generation.
    """

    @staticmethod
    def create_preprocessor_for_dataframe(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        scaling_method: str = "standard",
        encoding_method: str = "onehot"
    ) -> Tuple[ColumnTransformer, List[str], List[str]]:
        """
        Infers feature types from DataFrame and constructs ColumnTransformer.
        """
        feature_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df.copy()

        num_cols = list(feature_df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(feature_df.select_dtypes(include=['object', 'category', 'bool']).columns)

        column_transformer = PreprocessingPipelineSuite.build_column_transformer(
            numerical_features=num_cols,
            categorical_features=cat_cols,
            scaling_method=scaling_method,
            encoding_method=encoding_method
        )

        return column_transformer, num_cols, cat_cols
