"""
MLForge - Monitoring & Drift Routes Blueprint
Renders model health dashboards, drift detection triggers, and retraining actions.
"""

from flask import Blueprint, render_template, jsonify
from services.monitoring_service import MonitoringService
from services.model_service import ModelService

monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")


@monitoring_bp.route("/")
def index():
    """
    Renders model health monitoring dashboard and drift report list.
    """
    summary = MonitoringService.get_dashboard_summary()
    drift_reports = MonitoringService.list_drift_reports()
    models = ModelService.list_models()
    
    return render_template(
        "monitoring/index.html",
        summary=summary,
        drift_reports=drift_reports,
        models=models,
        active_tab="monitoring"
    )


@monitoring_bp.route("/api/reports")
def api_reports():
    """Returns JSON list of data drift reports."""
    return jsonify({"status": "success", "reports": MonitoringService.list_drift_reports()})
