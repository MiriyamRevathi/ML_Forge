"""
MLForge - System Monitoring & Health Service Module
Aggregates platform statistics, computes overall model health scores,
manages drift reports, and stores monitoring baselines.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from config import MONITORING_DIR
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.experiment_service import ExperimentService
from services.pipeline_service import PipelineService
from services.prediction_service import PredictionService
from utils.files import load_json, save_json, list_files_in_dir
from utils.helpers import get_current_timestamp_iso


class MonitoringService:
    """
    Service for calculating platform health metrics, monitoring dashboards, and drift alert summaries.
    """
    
    @staticmethod
    def get_dashboard_summary() -> Dict[str, Any]:
        """
        Calculates top-level summary metrics for the primary ML Systems Dashboard.
        """
        datasets = DatasetService.list_datasets()
        experiments = ExperimentService.list_experiments()
        pipeline_runs = PipelineService.list_pipeline_runs()
        models = ModelService.list_models()
        prod_models = [m for m in models if m.get("status") == "PRODUCTION"]
        predictions = PredictionService.list_predictions()
        drift_reports = MonitoringService.list_drift_reports()
        
        # Calculate active drift alerts count
        drift_alerts_count = 0
        for r in drift_reports:
            if r.get("has_drift") or r.get("drift_status") == "WARNING" or r.get("drift_status") == "DRIFT DETECTED":
                drift_alerts_count += 1
                
        # Overall model health score calculation (0 to 100)
        health_score = 100
        if not prod_models:
            health_score -= 15
        if drift_alerts_count > 0:
            health_score -= (drift_alerts_count * 10)
        health_score = max(min(health_score, 100), 40)

        health_status = "HEALTHY"
        if health_score < 70:
            health_status = "CRITICAL"
        elif health_score < 85:
            health_status = "WARNING"

        return {
            "datasets_count": len(datasets),
            "experiments_count": len(experiments),
            "pipeline_runs_count": len(pipeline_runs),
            "models_count": len(models),
            "production_models_count": len(prod_models),
            "predictions_count": len(predictions),
            "drift_alerts_count": drift_alerts_count,
            "health_score": health_score,
            "health_status": health_status,
            "recent_activity": (experiments[:5] if experiments else [])
        }

    @staticmethod
    def list_drift_reports() -> List[Dict[str, Any]]:
        """
        Lists stored data drift reports.
        """
        report_files = list_files_in_dir(MONITORING_DIR, extension="json")
        reports = []
        
        for rf in report_files:
            if rf.name.startswith("drift_"):
                try:
                    rep_data = load_json(rf)
                    reports.append(rep_data)
                except Exception:
                    continue
                    
        return sorted(reports, key=lambda x: x.get("timestamp", ""), reverse=True)

    @staticmethod
    def save_drift_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves a newly computed data drift analysis report.
        """
        report_id = report_data.get("id") or get_current_timestamp_iso().replace(":", "-")
        report_data["id"] = report_id
        
        if "timestamp" not in report_data:
            report_data["timestamp"] = get_current_timestamp_iso()
            
        report_path = MONITORING_DIR / f"drift_{report_id}.json"
        save_json(report_data, report_path)
        return report_data
