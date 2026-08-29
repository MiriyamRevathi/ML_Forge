"""
MLForge Services - Platform Reporting Service Module
Dispatches Markdown and JSON report generation for quality audits, EDA reports,
pipeline run logs, model comparison matrices, and statistical data drift reports.
"""

from typing import Dict, List, Any, Optional
from services.dataset_service import DatasetService
from services.pipeline_service import PipelineService
from services.monitoring_service import MonitoringService
from ml.reports_generator import ReportGenerator
from ml.quality_engine import DataQualityEngine
from ml.quality_scoring import QualityScorer


class ReportingService:
    """
    Business logic service for generating platform reports.
    """

    @staticmethod
    def get_quality_report_markdown(dataset_id: str) -> str:
        """
        Compiles Data Quality Markdown Report.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            return "# Error\nDataset not found."

        df = DatasetService.load_dataset_dataframe(dataset_id)
        engine = DataQualityEngine(df, target_column=meta.get("target_column"))
        quality_data = engine.run_all_quality_checks()
        quality_data["dataset"] = meta

        return ReportGenerator.generate_data_quality_report_md(quality_data)

    @staticmethod
    def get_pipeline_run_markdown(run_id: str) -> str:
        """
        Compiles Pipeline Run Execution Markdown Report.
        """
        run_data = PipelineService.get_pipeline_run(run_id)
        if not run_data:
            return "# Error\nPipeline run record not found."

        return ReportGenerator.generate_pipeline_run_report_md(run_data)

    @staticmethod
    def get_drift_report_markdown(report_id: str) -> str:
        """
        Compiles Data Drift Markdown Report.
        """
        reports = MonitoringService.list_drift_reports()
        match = next((r for r in reports if r.get("id") == report_id), None)
        if not match:
            return "# Error\nDrift report record not found."

        return ReportGenerator.generate_drift_report_md(match)
