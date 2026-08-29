"""
MLForge ML Engine - EDA Engine Suite Module
Manages dataset statistical profiling, distribution analysis, and Matplotlib chart generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.exploration import ExploratoryDataAnalysis


class EDAEngineSuite:
    """
    Business logic suite for EDA reports and visualizations.
    """

    @staticmethod
    def get_numerical_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        return ExploratoryDataAnalysis.get_numerical_statistics(df)

    @staticmethod
    def get_categorical_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        return ExploratoryDataAnalysis.get_categorical_statistics(df)

    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        return ExploratoryDataAnalysis.get_correlation_matrix(df)

    @staticmethod
    def create_distribution_chart(df: pd.DataFrame) -> Optional[str]:
        return ExploratoryDataAnalysis.create_distribution_chart(df)

    @staticmethod
    def create_box_plot_chart(df: pd.DataFrame) -> Optional[str]:
        return ExploratoryDataAnalysis.create_box_plot_chart(df)

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> Optional[str]:
        return ExploratoryDataAnalysis.create_correlation_heatmap(df)

    @staticmethod
    def create_missing_values_chart(df: pd.DataFrame) -> Optional[str]:
        return ExploratoryDataAnalysis.create_missing_values_chart(df)

    @staticmethod
    def create_target_distribution_chart(df: pd.DataFrame, target_column: str) -> Optional[str]:
        return ExploratoryDataAnalysis.create_target_distribution_chart(df, target_column)
