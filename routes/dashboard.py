"""
MLForge - Dashboard Route Blueprint
Renders the primary ML Systems Dashboard and provides quick summary JSON APIs.
"""

from flask import Blueprint, render_template, jsonify
from services.monitoring_service import MonitoringService
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.experiment_service import ExperimentService

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """
    Renders the main MLForge Systems Dashboard view.
    """
    summary = MonitoringService.get_dashboard_summary()
    datasets = DatasetService.list_datasets()[:5]
    models = ModelService.list_models()[:5]
    experiments = ExperimentService.list_experiments()[:5]
    
    return render_template(
        "dashboard.html",
        summary=summary,
        datasets=datasets,
        models=models,
        experiments=experiments,
        active_tab="dashboard"
    )


@dashboard_bp.route("/api/dashboard/summary")
def api_summary():
    """
    API endpoint returning live dashboard summary statistics as JSON.
    """
    summary = MonitoringService.get_dashboard_summary()
    return jsonify({"status": "success", "data": summary})
