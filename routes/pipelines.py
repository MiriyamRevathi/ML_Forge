"""
MLForge - Pipeline Routes Blueprint
Renders the visual Pipeline Builder DAG interface and pipeline execution triggers.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from services.pipeline_service import PipelineService
from services.dataset_service import DatasetService
from config import CLASSIFICATION_MODELS, REGRESSION_MODELS

pipelines_bp = Blueprint("pipelines", __name__, url_prefix="/pipelines")


@pipelines_bp.route("/")
def index():
    """
    Renders visual Pipeline Builder interface.
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


@pipelines_bp.route("/api/save", methods=["POST"])
def save_pipeline_api():
    """
    API endpoint to save or update pipeline JSON configuration.
    """
    data = request.json or {}
    saved = PipelineService.save_pipeline(data)
    return jsonify({"status": "success", "pipeline": saved})


@pipelines_bp.route("/api/list")
def list_pipelines_api():
    """API endpoint to list pipelines."""
    return jsonify({"status": "success", "pipelines": PipelineService.list_pipelines()})
