"""
MLForge - Dataset Routes Blueprint
Provides HTTP endpoints for dataset upload, inspection, preview, validation audit,
Exploratory Data Analysis (EDA) charts, metadata, and deletion.
"""

import os
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from services.dataset_service import DatasetService
from utils.files import sanitize_filename
from utils.validation import validate_dataset_upload_request, ValidationError

datasets_bp = Blueprint("datasets", __name__, url_prefix="/datasets")


@datasets_bp.route("/")
def index():
    """
    Renders dataset management dashboard listing all datasets.
    """
    datasets = DatasetService.list_datasets()
    return render_template("datasets/index.html", datasets=datasets, active_tab="datasets")


@datasets_bp.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handles CSV dataset upload from HTTP form.
    """
    if "file" not in request.files:
        flash("No file part provided in upload request.", "danger")
        return redirect(url_for("datasets.index"))
        
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected for upload.", "warning")
        return redirect(url_for("datasets.index"))
        
    try:
        filename = sanitize_filename(file.filename)
        custom_name = request.form.get("dataset_name", "").strip() or None
        target_column = request.form.get("target_column", "").strip() or None
        task_type = request.form.get("task_type", "classification")
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        metadata = DatasetService.register_dataset(
            filepath=Path(tmp_path),
            custom_name=custom_name,
            target_column=target_column,
            task_type=task_type
        )
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        flash(f"Dataset '{metadata['name']}' uploaded successfully!", "success")
    except ValidationError as ve:
        flash(f"Validation Error: {str(ve)}", "danger")
    except Exception as e:
        flash(f"Failed to process uploaded file: {str(e)}", "danger")
        
    return redirect(url_for("datasets.index"))


@datasets_bp.route("/<dataset_id>")
def detail(dataset_id):
    """
    Shows detailed view and schema summary for a specific dataset.
    """
    metadata = DatasetService.get_dataset_metadata(dataset_id)
    if not metadata:
        flash(f"Dataset ID '{dataset_id}' not found.", "danger")
        return redirect(url_for("datasets.index"))
        
    preview_data = []
    try:
        df = DatasetService.load_dataset_dataframe(dataset_id, max_rows=10)
        preview_data = df.to_dict(orient="records")
    except Exception as e:
        flash(f"Warning: Could not read dataset preview: {e}", "warning")
        
    return render_template(
        "datasets/detail.html",
        dataset=metadata,
        preview_data=preview_data,
        active_tab="datasets"
    )


@datasets_bp.route("/<dataset_id>/validation")
def validation(dataset_id):
    """
    Runs automated data quality & validation suite and renders report.
    """
    try:
        report = DatasetService.validate_dataset(dataset_id)
        return render_template(
            "datasets/validation.html",
            report=report,
            dataset=report["dataset"],
            active_tab="datasets"
        )
    except Exception as e:
        flash(f"Data validation failed: {str(e)}", "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))


@datasets_bp.route("/<dataset_id>/eda")
def eda(dataset_id):
    """
    Executes Exploratory Data Analysis (EDA) and renders interactive chart report.
    """
    try:
        eda_data = DatasetService.run_eda(dataset_id)
        return render_template(
            "datasets/eda.html",
            eda=eda_data,
            dataset=eda_data["dataset"],
            active_tab="datasets"
        )
    except Exception as e:
        flash(f"EDA generation failed: {str(e)}", "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))


@datasets_bp.route("/<dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id):
    """
    Deletes a dataset by ID.
    """
    success = DatasetService.delete_dataset(dataset_id)
    if success:
        flash(f"Dataset deleted successfully.", "info")
    else:
        flash(f"Failed to delete dataset '{dataset_id}'.", "danger")
        
    return redirect(url_for("datasets.index"))


@datasets_bp.route("/api/list")
def api_list():
    """Returns JSON list of all available datasets."""
    return jsonify({"status": "success", "datasets": DatasetService.list_datasets()})


@datasets_bp.route("/api/<dataset_id>/eda")
def api_eda(dataset_id):
    """Returns JSON response of EDA statistics and charts."""
    try:
        eda_data = DatasetService.run_eda(dataset_id)
        return jsonify({"status": "success", "data": eda_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
