"""
MLForge - Monitoring & Drift Routes Blueprint
Renders model health dashboards, statistical data drift analysis triggers, and automated retraining endpoints.
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from services.monitoring_service import MonitoringService
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.pipeline_service import PipelineService
from ml.drift import DataDriftDetector
from ml.retraining import ModelRetrainer

monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")


@monitoring_bp.route("/")
def index():
    """
    Renders model health monitoring dashboard and drift report list.
    """
    summary = MonitoringService.get_dashboard_summary()
    drift_reports = MonitoringService.list_drift_reports()
    datasets = DatasetService.list_datasets()
    pipelines = PipelineService.list_pipelines()
    models = ModelService.list_models()
    
    return render_template(
        "monitoring/index.html",
        summary=summary,
        drift_reports=drift_reports,
        datasets=datasets,
        pipelines=pipelines,
        models=models,
        active_tab="monitoring"
    )


@monitoring_bp.route("/run_drift", methods=["POST"])
def run_drift_analysis():
    """
    Triggers statistical data drift analysis comparing reference and target datasets.
    """
    ref_ds = request.form.get("reference_dataset")
    tar_ds = request.form.get("target_dataset")
    
    if not ref_ds or not tar_ds:
        flash("Please select both a reference dataset and a target dataset for drift analysis.", "warning")
        return redirect(url_for("monitoring.index"))
        
    try:
        report = DataDriftDetector.detect_drift(ref_ds, tar_ds)
        if report["has_drift"]:
            flash(f"Drift Analysis Completed: DRIFT DETECTED across {report['drifted_features_count']} features!", "warning")
        else:
            flash(f"Drift Analysis Completed: Feature distributions are NORMAL.", "success")
    except Exception as e:
        flash(f"Drift analysis failed: {str(e)}", "danger")
        
    return redirect(url_for("monitoring.index"))


@monitoring_bp.route("/retrain", methods=["POST"])
def retrain_model_action():
    """
    Triggers automated model retraining on new data and auto-promotes if performance improves.
    """
    pipeline_id = request.form.get("pipeline_id")
    target_dataset = request.form.get("target_dataset")
    
    if not pipeline_id:
        flash("Please select a pipeline configuration for retraining.", "warning")
        return redirect(url_for("monitoring.index"))
        
    try:
        result = ModelRetrainer.retrain_model(pipeline_id, target_dataset_id=target_dataset)
        if result["promoted_to_production"]:
            flash(f"Retraining Complete! New model '{result['candidate_version']}' PROMOTED to PRODUCTION! Reason: {result['promotion_reason']}", "success")
        else:
            flash(f"Retraining Complete! Candidate model '{result['candidate_version']}' kept in STAGING. Reason: {result['promotion_reason']}", "info")
    except Exception as e:
        flash(f"Retraining failed: {str(e)}", "danger")
        
    return redirect(url_for("monitoring.index"))


@monitoring_bp.route("/api/reports")
def api_reports():
    """Returns JSON list of data drift reports."""
    return jsonify({"status": "success", "reports": MonitoringService.list_drift_reports()})
