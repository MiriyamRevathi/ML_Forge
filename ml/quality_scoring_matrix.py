"""
MLForge ML Engine - Data Quality Scoring Matrix & Grading Engine Module
Calculates weighted quality dimensional matrices (Completeness, Consistency, Validity, Uniqueness, Balance),
computes overall quality index (0-100), assigns letter grades, and produces remediation action lists.
"""

from typing import Dict, List, Any, Tuple, Optional


class QualityScoringMatrix:
    """
    Data Quality Dimensional Matrix & Scoring Engine.
    """

    DIMENSION_WEIGHTS = {
        "completeness": 0.25,
        "validity": 0.25,
        "uniqueness": 0.20,
        "consistency": 0.20,
        "balance": 0.10
    }

    @staticmethod
    def compute_dimensional_matrix(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes sub-scores for Completeness, Validity, Uniqueness, Consistency, Balance.
        """
        scores = {
            "completeness": 100.0,
            "validity": 100.0,
            "uniqueness": 100.0,
            "consistency": 100.0,
            "balance": 100.0
        }

        for check in checks:
            rule_id = check.get("rule_id", "")
            status = check.get("status", "PASS")
            penalty = 30.0 if status == "FAIL" else (12.0 if status == "WARN" else 0.0)

            if "MISSING" in rule_id or "EMPTY" in rule_id:
                scores["completeness"] = max(scores["completeness"] - penalty, 0.0)
            elif "DUPLICATE" in rule_id:
                scores["uniqueness"] = max(scores["uniqueness"] - penalty, 0.0)
            elif "INFINITE" in rule_id or "OUTLIERS" in rule_id or "CARDINALITY" in rule_id:
                scores["validity"] = max(scores["validity"] - penalty, 0.0)
            elif "CONSTANT" in rule_id or "LEAKAGE" in rule_id:
                scores["consistency"] = max(scores["consistency"] - penalty, 0.0)
            elif "IMBALANCE" in rule_id:
                scores["balance"] = max(scores["balance"] - penalty, 0.0)

        # Calculate weighted overall index
        overall_index = round(
            sum(scores[dim] * weight for dim, weight in QualityScoringMatrix.DIMENSION_WEIGHTS.items()),
            1
        )

        grade, badge_class = QualityScoringMatrix._grade_score(overall_index)

        return {
            "overall_quality_index": overall_index,
            "grade": grade,
            "badge_class": badge_class,
            "dimension_scores": {k: round(v, 1) for k, v in scores.items()},
            "weights": QualityScoringMatrix.DIMENSION_WEIGHTS
        }

    @staticmethod
    def _grade_score(score: float) -> Tuple[str, str]:
        if score >= 95.0:
            return "A+", "badge-success"
        elif score >= 88.0:
            return "A", "badge-success"
        elif score >= 75.0:
            return "B", "badge-info"
        elif score >= 60.0:
            return "C", "badge-warning"
        elif score >= 45.0:
            return "D", "badge-warning"
        else:
            return "F", "badge-danger"
