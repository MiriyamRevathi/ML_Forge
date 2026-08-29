"""
MLForge ML Engine - Feature Engineering Suite Facade Module
High-level facade for applying mathematical log/sqrt/power transforms, ratio features,
interaction products, datetime component extractors, and feature selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.feature_transformations_suite import FeatureTransformationsSuite


class FeatureEngineeringSuiteFacade:
    """
    High-level facade for Feature Engineering workflows.
    """

    @staticmethod
    def execute_feature_engineering_pipeline(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        enable_log_transforms: bool = False,
        enable_sqrt_transforms: bool = False,
        enable_ratios: bool = False,
        enable_interactions: bool = False,
        filter_low_variance: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Executes multi-step feature engineering pipeline.
        """
        df_feat = df.copy()
        generated_features = []

        feature_cols = [c for c in df_feat.columns if c != target_column] if target_column else list(df_feat.columns)
        num_cols = list(df_feat[feature_cols].select_dtypes(include=[np.number]).columns)

        if filter_low_variance and num_cols:
            df_feat, dropped = FeatureTransformationsSuite.filter_low_variance_features(df_feat, threshold=0.0)

        if enable_log_transforms and num_cols:
            df_feat, log_cols = FeatureTransformationsSuite.apply_log_transform(df_feat, columns=num_cols)
            generated_features.extend(log_cols)

        if enable_sqrt_transforms and num_cols:
            df_feat, sqrt_cols = FeatureTransformationsSuite.apply_sqrt_transform(df_feat, columns=num_cols)
            generated_features.extend(sqrt_cols)

        if enable_ratios and len(num_cols) >= 2:
            pairs = [(num_cols[i], num_cols[i+1]) for i in range(len(num_cols)-1)]
            df_feat, ratio_cols = FeatureTransformationsSuite.create_ratio_features(df_feat, feature_pairs=pairs)
            generated_features.extend(ratio_cols)

        return df_feat, generated_features
