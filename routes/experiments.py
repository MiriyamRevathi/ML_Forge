"""
MLForge - Experiment Tracking Routes Blueprint
Renders experiment history, metrics comparison views, and parameters log.
"""

from flask import Blueprint, render_template, jsonify
from services.experiment_service import ExperimentService

experiments_bp = Blueprint("experiments", __name__, url_prefix="/experiments")


@experiments_bp.route("/")
def index():
    """
    Renders experiment run history dashboard.
    """
    experiments = ExperimentService.list_experiments()
    return render_template(
        "experiments/index.html",
        experiments=experiments,
        active_tab="experiments"
    )


@experiments_bp.route("/<experiment_id>")
def detail(experiment_id):
    """
    Shows detail metrics and hyperparameters for an experiment run.
    """
    exp = ExperimentService.get_experiment(experiment_id)
    return render_template(
        "experiments/detail.html",
        experiment=exp,
        active_tab="experiments"
    )


@experiments_bp.route("/api/list")
def api_list():
    """Returns JSON list of experiment records."""
    return jsonify({"status": "success", "experiments": ExperimentService.list_experiments()})
