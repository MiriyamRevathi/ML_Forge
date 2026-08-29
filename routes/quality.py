"""
MLForge - Quality Routes Blueprint
Provides HTTP views and JSON APIs for Data Quality Audit reports, quality dimension scores,
and remediation action items.
"""

from flask import Blueprint, render_template, jsonify, flash, redirect, url_for
from services.dataset_service import DatasetService
from services.quality_service import QualityService

quality_bp = Blueprint("quality", __name__, url_prefix="/quality")


@quality_bp.route("/")
def index():
    """
    Renders Data Quality Audit index page listing all datasets.
    """
    datasets = DatasetService.list_datasets()
    return render_template(
        "quality/index.html",
        datasets=datasets,
        active_tab="quality"
    )


@quality_bp.route("/<dataset_id>")
def audit_detail(dataset_id):
    """
    Renders detailed Data Quality Audit report for a dataset.
    """
    try:
        audit_data = QualityService.audit_dataset_quality(dataset_id)
        return render_template(
            "quality/detail.html",
            audit=audit_data,
            dataset=audit_data["dataset"],
            quality=audit_data["quality_report"],
            scoring=audit_data["scoring"],
            active_tab="quality"
        )
    except Exception as e:
        flash(f"Failed to load quality audit: {str(e)}", "danger")
        return redirect(url_for("quality.index"))


@quality_bp.route("/api/<dataset_id>")
def api_audit(dataset_id):
    """Returns JSON payload of data quality audit."""
    try:
        audit_data = QualityService.audit_dataset_quality(dataset_id)
        return jsonify({"status": "success", "data": audit_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
