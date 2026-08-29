"""
MLForge - Dataset Routes Blueprint
Provides HTTP endpoints for dataset upload, dataset inspection, preview,
validation, metadata retrieval, and dataset deletion.
"""

import os
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
        
        # Save temp file
        temp_path = DatasetService.get_dataset_csv_path("temp") or (DatasetService.list_datasets()[0] if DatasetService.list_datasets() else None)
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
        
        # Clean temp
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
        df = DatasetService.load_dataset_dataframe(dataset_id)
        preview_data = df.head(10).to_dict(orient="records")
    except Exception as e:
        flash(f"Warning: Could not read dataset preview: {e}", "warning")
        
    return render_template(
        "datasets/detail.html",
        dataset=metadata,
        preview_data=preview_data,
        active_tab="datasets"
    )


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
