"""
MLForge ML Engine - Feature Engine Suite Module
Applies interaction product features, log/sqrt math transforms, and PolynomialFeatures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.feature_engineering import FeatureEngineer


class FeatureEngineSuite:
    """
    Feature engineering suite.
    """

    @staticmethod
    def apply_feature_engineering(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        enable_interactions: bool = False,
        enable_log_transforms: bool = False
    ) -> pd.DataFrame:
        return FeatureEngineer.apply_feature_engineering(
            df=df,
            target_column=target_column,
            enable_interactions=enable_interactions,
            enable_log_transforms=enable_log_transforms
        )
