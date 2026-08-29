"""
MLForge ML Engine - EDA Statistical Reporting Suite Module
Compiles multi-page Exploratory Data Analysis summaries, distribution metrics,
quantile matrices, frequency tables, and correlation report payloads.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ml.eda_statistics import EDAStatisticsCalculator
from ml.eda_visualizer import EDAVisualizer


class EDAReportingSuite:
    """
    Exploratory Data Analysis Reporting & Metrics Suite.
    """

    @staticmethod
    def compile_eda_suite(df: pd.DataFrame, dataset_name: str = "Dataset", target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Compiles complete EDA statistical report and chart data payload.
        """
        num_summaries = EDAStatisticsCalculator.calculate_numerical_summaries(df)
        cat_summaries = EDAStatisticsCalculator.calculate_categorical_summaries(df)
        bivariate_corrs = EDAStatisticsCalculator.calculate_bivariate_correlations(df)

        charts = {
            "histograms": EDAVisualizer.render_histogram_grid(df),
            "boxplots": EDAVisualizer.render_boxplot_grid(df),
            "correlation_heatmap": EDAVisualizer.render_correlation_heatmap(df),
            "missing_values_bar": EDAVisualizer.render_missing_values_bar(df)
        }

        return {
            "dataset_name": dataset_name,
            "target_column": target_column,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_summaries": num_summaries,
            "categorical_summaries": cat_summaries,
            "bivariate_correlations": bivariate_corrs,
            "charts": charts
        }
