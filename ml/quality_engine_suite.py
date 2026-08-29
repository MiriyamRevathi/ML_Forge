"""
MLForge Data Quality Engine - Quality Engine Suite Module
Audits missing values, duplicates, constant/near-constant features, infinite values,
extreme outliers, target leakage indicators, class imbalance, and data type inconsistencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.quality_engine import DataQualityEngine


class QualityEngineSuite:
    """
    Advanced Data Quality & Integrity Inspection Engine Suite.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.engine = DataQualityEngine(df, target_column=target_column)

    def run_all_quality_checks(self) -> Dict[str, Any]:
        return self.engine.run_all_quality_checks()
