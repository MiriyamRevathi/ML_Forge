"""
MLForge - Pipeline Routes Blueprint
Renders visual Pipeline Builder DAG interface, pipeline execution triggers,
run log inspector, and JSON API execution endpoints.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from services.pipeline_service import PipelineService
from services.dataset_service import DatasetService
from ml.pipeline import PipelineEngine
from config import CLASSIFICATION_MODELS, REGRESSION_MODELS

pipelines_bp = Blueprint("pipelines", __name__, url_prefix="/pipelines")


@pipelines_bp.route("/")
def index():
    """
    Renders visual Pipeline Builder interface and execution history list.
    """
    pipelines = PipelineService.list_pipelines()
    runs = PipelineService.list_pipeline_runs()
    datasets = DatasetService.list_datasets()
    
    return render_template(
        "pipelines/index.html",
        pipelines=pipelines,
        runs=runs,
        datasets=datasets,
        classification_models=CLASSIFICATION_MODELS,
        regression_models=REGRESSION_MODELS,
        active_tab="pipelines"
    )


@pipelines_bp.route("/builder")
def builder():
    """
    Renders interactive pipeline DAG node configuration canvas.
    """
    datasets = DatasetService.list_datasets()
    return render_template(
        "pipelines/builder.html",
        datasets=datasets,
        classification_models=CLASSIFICATION_MODELS,
        regression_models=REGRESSION_MODELS,
        active_tab="pipelines"
    )


@pipelines_bp.route("/<pipeline_id>/run", methods=["POST"])
def run_pipeline_http(pipeline_id):
    """
    HTTP POST trigger to execute a saved pipeline by ID.
    """
    pipeline_cfg = PipelineService.get_pipeline(pipeline_id)
    if not pipeline_cfg:
        flash(f"Pipeline ID '{pipeline_id}' not found.", "danger")
        return redirect(url_for("pipelines.index"))
        
    try:
        engine = PipelineEngine(pipeline_cfg)
        run_result = engine.execute()
        flash(f"Pipeline '{pipeline_cfg.get('name')}' executed successfully! Model version '{run_result['model_version']}' created.", "success")
        return redirect(url_for("pipelines.run_detail", run_id=run_result["run_id"]))
    except Exception as e:
        flash(f"Pipeline execution failed: {str(e)}", "danger")
        return redirect(url_for("pipelines.index"))


@pipelines_bp.route("/runs/<run_id>")
def run_detail(run_id):
    """
    Renders detailed execution log output and metric summary for a specific pipeline run.
    """
    run_record = PipelineService.get_pipeline_run(run_id)
    if not run_record:
        flash(f"Pipeline run ID '{run_id}' not found.", "danger")
        return redirect(url_for("pipelines.index"))
        
    return render_template(
        "pipelines/run_detail.html",
        run=run_record,
        active_tab="pipelines"
    )


@pipelines_bp.route("/api/save", methods=["POST"])
def save_pipeline_api():
    """
    API endpoint to save or update pipeline JSON configuration.
    """
    data = request.json or {}
    saved = PipelineService.save_pipeline(data)
    return jsonify({"status": "success", "pipeline": saved})


@pipelines_bp.route("/api/run", methods=["POST"])
def run_pipeline_api():
    """
    API endpoint to execute pipeline DAG in real-time.
    """
    data = request.json or {}
    pipeline_id = data.get("pipeline_id")
    
    if pipeline_id:
        pipeline_cfg = PipelineService.get_pipeline(pipeline_id)
    else:
        pipeline_cfg = data.get("config")
        
    if not pipeline_cfg:
        return jsonify({"status": "error", "message": "Missing pipeline configuration."}), 400
        
    try:
        engine = PipelineEngine(pipeline_cfg)
        run_result = engine.execute()
        return jsonify({"status": "success", "run": run_result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@pipelines_bp.route("/api/list")
def list_pipelines_api():
    """API endpoint to list pipelines."""
    return jsonify({"status": "success", "pipelines": PipelineService.list_pipelines()})
