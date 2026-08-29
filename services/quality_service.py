"""
MLForge Services - Data Quality Service Module
Manages quality audits, quality scoring, dimension metrics, and remediation recommendation dispatch.
"""

from typing import Dict, List, Any, Optional
from services.dataset_service import DatasetService
from ml.quality_engine import DataQualityEngine
from ml.quality_scoring import QualityScorer


class QualityService:
    """
    Business logic service for data quality audits.
    """

    @staticmethod
    def audit_dataset_quality(dataset_id: str) -> Dict[str, Any]:
        """
        Runs quality audit engine and calculates dimensional quality score breakdown.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            raise FileNotFoundError(f"Dataset ID '{dataset_id}' not found.")

        df = DatasetService.load_dataset_dataframe(dataset_id)
        target_col = meta.get("target_column")

        engine = DataQualityEngine(df, target_column=target_col)
        quality_report = engine.run_all_quality_checks()

        dimensions_summary = QualityScorer.calculate_quality_dimensions(quality_report)

        return {
            "dataset": meta,
            "quality_report": quality_report,
            "scoring": dimensions_summary
        }
