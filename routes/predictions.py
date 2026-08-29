"""
MLForge - Prediction Routes Blueprint
Renders online inference forms, single prediction UI, and CSV batch prediction UI.
"""

from flask import Blueprint, render_template, request, jsonify
from services.model_service import ModelService
from services.prediction_service import PredictionService

predictions_bp = Blueprint("predictions", __name__, url_prefix="/predictions")


@predictions_bp.route("/")
def index():
    """
    Renders prediction workspace for single & batch prediction.
    """
    models = ModelService.list_models()
    prod_model = ModelService.get_production_model()
    past_predictions = PredictionService.list_predictions()[:10]
    
    return render_template(
        "predictions/index.html",
        models=models,
        production_model=prod_model,
        past_predictions=past_predictions,
        active_tab="predictions"
    )


@predictions_bp.route("/api/list")
def api_list():
    """Returns JSON list of prediction logs."""
    return jsonify({"status": "success", "predictions": PredictionService.list_predictions()})
