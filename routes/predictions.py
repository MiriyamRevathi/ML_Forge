"""
MLForge - Prediction Routes Blueprint
Renders online inference forms, single prediction UI, CSV batch prediction UI,
dynamic feature schema generator, and CSV download endpoints.
"""

from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from services.model_service import ModelService
from services.prediction_service import PredictionService
from ml.prediction import SinglePredictor
from ml.batch_prediction import BatchPredictor
from config import PREDICTION_DIR
from utils.files import sanitize_filename

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


@predictions_bp.route("/api/schema/<model_version>")
def api_schema(model_version):
    """
    Returns feature names list and metadata for generating dynamic form fields in UI.
    """
    meta = ModelService.get_model_metadata(model_version)
    if not meta:
        return jsonify({"status": "error", "message": "Model version not found"}), 404
        
    return jsonify({
        "status": "success",
        "version": model_version,
        "name": meta.get("name"),
        "task_type": meta.get("task_type"),
        "feature_names": meta.get("feature_names", []),
        "target_column": meta.get("target_column")
    })


@predictions_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for single online inference request.
    """
    data = request.json or {}
    model_version = data.get("model_version")
    input_data = data.get("input_data", {})
    
    if not model_version or not input_data:
        return jsonify({"status": "error", "message": "Missing model_version or input_data payload."}), 400
        
    try:
        result = SinglePredictor.predict(model_version, input_data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@predictions_bp.route("/api/batch", methods=["POST"])
def api_batch():
    """
    API endpoint for bulk CSV batch prediction request.
    """
    if "batch_file" not in request.files:
        return jsonify({"status": "error", "message": "No batch_file provided."}), 400
        
    file = request.files["batch_file"]
    model_version = request.form.get("model_version")
    
    if not file or not model_version:
        return jsonify({"status": "error", "message": "Missing file or model_version parameter."}), 400
        
    try:
        import tempfile
        filename = sanitize_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        result = BatchPredictor.process_batch_csv(
            model_version=model_version,
            input_csv_path=Path(tmp_path),
            original_filename=filename
        )
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@predictions_bp.route("/download/<filename>")
def download_predictions(filename):
    """
    CSV file download route for batch predictions.
    """
    safe_filename = sanitize_filename(filename)
    return send_from_directory(PREDICTION_DIR, safe_filename, as_attachment=True)


@predictions_bp.route("/api/list")
def api_list():
    """Returns JSON list of prediction logs."""
    return jsonify({"status": "success", "predictions": PredictionService.list_predictions()})
