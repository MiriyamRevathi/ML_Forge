"""
MLForge Data Quality Engine - Quality Scoring Suite Module
Calculates weighted sub-scores (Completeness, Consistency, Validity, Uniqueness, Balance)
and assigns letter grades (A+, A, B, C, D, F) with remediation action items.
"""

from typing import Dict, List, Any, Optional
from ml.quality_scoring import QualityScorer


class QualityScoringSuite:
    """
    Data quality scoring and grading calculator suite.
    """

    @staticmethod
    def calculate_quality_dimensions(quality_report: Dict[str, Any]) -> Dict[str, Any]:
        return QualityScorer.calculate_quality_dimensions(quality_report)
