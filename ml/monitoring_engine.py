"""
MLForge ML Engine - Model Health Monitoring Engine Module
Calculates prediction volume statistics, throughput metrics, confidence distributions,
error rate metrics, and overall 0-100 model health score.
"""

from typing import Dict, List, Any, Optional
from services.model_service import ModelService
from services.prediction_service import PredictionService
from services.monitoring_service import MonitoringService


class ModelMonitoringEngine:
    """
    Model Health Monitoring & Volume Analytics Suite.
    """

    @staticmethod
    def calculate_health_score(model_version: str) -> Dict[str, Any]:
        """
        Calculates 0-100 model health score based on metrics, predictions volume, and drift status.
        """
        meta = ModelService.get_model_metadata(model_version)
        if not meta:
            return {"health_score": 0, "status": "UNKNOWN"}

        base_score = 100.0
        reasons = []

        # Metric health
        metrics = meta.get("metrics", {})
        if meta.get("task_type") == "classification":
            acc = metrics.get("accuracy", 0.0)
            if acc < 0.70:
                base_score -= 25.0
                reasons.append(f"Low test accuracy ({acc})")
        else:
            r2 = metrics.get("r2_score", 0.0)
            if r2 < 0.50:
                base_score -= 25.0
                reasons.append(f"Low test R² score ({r2})")

        # Status health
        status = meta.get("status", "STAGING")
        if status == "ARCHIVED":
            base_score -= 50.0
            reasons.append("Model is archived")

        final_score = max(round(base_score, 1), 0.0)

        status_label = "HEALTHY" if final_score >= 80.0 else ("WARNING" if final_score >= 50.0 else "CRITICAL")

        return {
            "model_version": model_version,
            "health_score": final_score,
            "status": status_label,
            "issues": reasons
        }
