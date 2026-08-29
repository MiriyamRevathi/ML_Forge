"""
MLForge Services - Exploratory Data Analysis Service Module
Manages dataset statistical profiling, distribution analysis, and Matplotlib chart generation.
"""

from typing import Dict, List, Any, Optional
from services.dataset_service import DatasetService
from ml.dataset_profiler import DatasetProfiler
from ml.exploration import ExploratoryDataAnalysis


class EDAService:
    """
    Business logic service for EDA reports and visualizations.
    """

    @staticmethod
    def generate_eda_report(dataset_id: str) -> Dict[str, Any]:
        """
        Runs complete EDA pipeline and returns statistical metrics and base64 chart URIs.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            raise FileNotFoundError(f"Dataset ID '{dataset_id}' not found.")

        df = DatasetService.load_dataset_dataframe(dataset_id)
        target_col = meta.get("target_column", "")

        profiler = DatasetProfiler(df, dataset_name=meta.get("name", "Dataset"))
        profile_data = profiler.generate_complete_profile()

        num_stats = ExploratoryDataAnalysis.get_numerical_statistics(df)
        cat_stats = ExploratoryDataAnalysis.get_categorical_statistics(df)
        corr_matrix = ExploratoryDataAnalysis.get_correlation_matrix(df)

        charts = {
            "distribution": ExploratoryDataAnalysis.create_distribution_chart(df),
            "boxplot": ExploratoryDataAnalysis.create_box_plot_chart(df),
            "correlation": ExploratoryDataAnalysis.create_correlation_heatmap(df),
            "missing_values": ExploratoryDataAnalysis.create_missing_values_chart(df),
            "target": ExploratoryDataAnalysis.create_target_distribution_chart(df, target_col)
        }

        return {
            "dataset": meta,
            "profile": profile_data,
            "numerical_statistics": num_stats,
            "categorical_statistics": cat_stats,
            "correlation_matrix": corr_matrix,
            "charts": charts
        }
