"""
MLForge - Platform Settings Blueprint
Renders platform configuration options, storage management, environment settings,
and local directory audit controls.
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from config import BASE_DIR, DATA_DIR, DATASET_DIR, MODEL_DIR, EXPERIMENT_DIR, PIPELINE_DIR, MONITORING_DIR
from utils.files import get_file_size_formatted

settings_bp = Blueprint("settings", __name__, url_prefix="/settings")


@settings_bp.route("/")
def index():
    """
    Renders Platform Settings dashboard.
    """
    storage_info = {
        "datasets": get_file_size_formatted(DATASET_DIR),
        "models": get_file_size_formatted(MODEL_DIR),
        "experiments": get_file_size_formatted(EXPERIMENT_DIR),
        "pipelines": get_file_size_formatted(PIPELINE_DIR),
        "monitoring": get_file_size_formatted(MONITORING_DIR),
        "total_data": get_file_size_formatted(DATA_DIR)
    }

    return render_template(
        "settings/index.html",
        storage_info=storage_info,
        active_tab="settings"
    )
