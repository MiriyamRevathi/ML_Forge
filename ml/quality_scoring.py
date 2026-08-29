"""
MLForge Data Quality Engine - Quality Scoring & Grading Module
Calculates weighted sub-scores (Completeness, Consistency, Validity, Uniqueness, Balance)
and assigns letter grades (A+, A, B, C, D, F) with remediation action items.
"""

from typing import Dict, List, Any, Tuple, Optional


class QualityScorer:
    """
    Data quality scoring and grading calculator.
    """

    @staticmethod
    def calculate_quality_dimensions(
        quality_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decomposes data quality into 5 standard dimensions: Completeness, Consistency, Validity, Uniqueness, Balance.
        """
        checks = quality_report.get("checks", [])
        
        # Sub-score initialization
        completeness = 100.0
        consistency = 100.0
        validity = 100.0
        uniqueness = 100.0
        balance = 100.0
        
        for c in checks:
            rule_id = c.get("rule_id", "")
            status = c.get("status", "PASS")
            penalty = 25.0 if status == "FAIL" else (10.0 if status == "WARN" else 0.0)
            
            if "MISSING" in rule_id or "EMPTY" in rule_id:
                completeness = max(completeness - penalty, 0.0)
            elif "DUPLICATE" in rule_id:
                uniqueness = max(uniqueness - penalty, 0.0)
            elif "INFINITE" in rule_id or "OUTLIERS" in rule_id or "CARDINALITY" in rule_id:
                validity = max(validity - penalty, 0.0)
            elif "CONSTANT" in rule_id or "LEAKAGE" in rule_id:
                consistency = max(consistency - penalty, 0.0)
            elif "IMBALANCE" in rule_id:
                balance = max(balance - penalty, 0.0)

        # Weighted final score
        weighted_score = round(
            (completeness * 0.25) +
            (validity * 0.25) +
            (uniqueness * 0.20) +
            (consistency * 0.20) +
            (balance * 0.10),
            1
        )

        grade, grade_badge = QualityScorer._assign_grade(weighted_score)
        remediations = QualityScorer._generate_remediations(checks)

        return {
            "weighted_score": weighted_score,
            "grade": grade,
            "grade_badge": grade_badge,
            "dimensions": {
                "completeness": round(completeness, 1),
                "validity": round(validity, 1),
                "uniqueness": round(uniqueness, 1),
                "consistency": round(consistency, 1),
                "balance": round(balance, 1)
            },
            "remediations": remediations
        }

    @staticmethod
    def _assign_grade(score: float) -> Tuple[str, str]:
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

    @staticmethod
    def _generate_remediations(checks: List[Dict[str, Any]]) -> List[str]:
        remediations = []
        for c in checks:
            if c.get("status") in ["WARN", "FAIL"]:
                rule_id = c.get("rule_id", "")
                if "MISSING" in rule_id:
                    remediations.append("Apply missing value imputation (Mean/Median for numeric, Mode for categorical).")
                elif "DUPLICATE_ROWS" in rule_id:
                    remediations.append("Execute duplicate row removal strategy.")
                elif "CONSTANT" in rule_id:
                    remediations.append("Drop zero-variance constant columns before feature scaling.")
                elif "LEAKAGE" in rule_id:
                    remediations.append("Inspect highly correlated features for potential target data leakage.")
                elif "IMBALANCE" in rule_id:
                    remediations.append("Consider class re-balancing (e.g. SMOTE or class weight adjustment).")
        return remediations
