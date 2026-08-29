"""
MLForge - Model Registry Routes Blueprint
Renders model registry lifecycle status board (TRAINED, STAGING, PRODUCTION, ARCHIVED),
model detail inspection, model comparison views, and state promotion endpoints.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from services.model_service import ModelService
from ml.comparison import ModelComparer
from ml.registry import ModelRegistryStateMachine

models_bp = Blueprint("models", __name__, url_prefix="/models")


@models_bp.route("/")
def index():
    """
    Renders Model Registry lifecycle dashboard.
    """
    models = ModelService.list_models()
    return render_template(
        "models/index.html",
        models=models,
        active_tab="models"
    )


@models_bp.route("/compare")
def compare():
    """
    Renders model evaluation comparison matrix and benchmark chart.
    """
    models = ModelService.list_models()
    comparison_results = ModelComparer.compare_models(models)
    
    return render_template(
        "models/compare.html",
        comparison=comparison_results,
        models=models,
        active_tab="models"
    )


@models_bp.route("/<model_version>")
def detail(model_version):
    """
    Renders detailed view for a model version.
    """
    model = ModelService.get_model_metadata(model_version)
    if not model:
        flash(f"Model version '{model_version}' not found.", "danger")
        return redirect(url_for("models.index"))
        
    return render_template(
        "models/detail.html",
        model=model,
        active_tab="models"
    )


@models_bp.route("/<model_version>/update_status", methods=["POST"])
def update_status(model_version):
    """
    Promotes or updates model status (e.g. TRAINED -> PRODUCTION).
    """
    new_status = request.form.get("status") or (request.json.get("status") if request.json else None)
    if not new_status:
        flash("No target status provided.", "warning")
        return redirect(url_for("models.index"))
        
    try:
        updated = ModelService.update_model_status(model_version, new_status)
        if updated:
            flash(f"Model version '{model_version}' status updated to {new_status}.", "success")
        else:
            flash(f"Model version '{model_version}' not found.", "danger")
    except Exception as e:
        flash(f"Error updating status: {str(e)}", "danger")
        
    return redirect(url_for("models.index"))


@models_bp.route("/api/list")
def api_list():
    """Returns JSON list of registered models."""
    return jsonify({"status": "success", "models": ModelService.list_models()})
