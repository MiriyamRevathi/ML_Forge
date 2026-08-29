"""JSON API for inspecting and controlling local MLForge jobs."""

from flask import Blueprint, jsonify, request

from services.job_service import (
    InvalidJobTransition,
    JobNotFoundError,
    JobService,
)

jobs_bp = Blueprint("jobs", __name__, url_prefix="/api/jobs")


@jobs_bp.errorhandler(JobNotFoundError)
def job_not_found(error):
    return jsonify({"status": "error", "message": "Job not found"}), 404


@jobs_bp.errorhandler(InvalidJobTransition)
def invalid_job_request(error):
    return jsonify({"status": "error", "message": str(error)}), 400


@jobs_bp.errorhandler(ValueError)
def invalid_value_request(error):
    return jsonify({"status": "error", "message": str(error)}), 400


@jobs_bp.route("", methods=["GET"])
def list_jobs():
    """List jobs with optional state and type filters."""
    state_values = request.args.getlist("state")
    job_type = request.args.get("type")
    jobs = JobService.list(states=state_values or None, job_type=job_type)
    return jsonify({"status": "success", "jobs": jobs, "count": len(jobs)})


@jobs_bp.route("/<job_id>", methods=["GET"])
def get_job(job_id):
    """Return one durable job record."""
    return jsonify({"status": "success", "job": JobService.get(job_id)})


@jobs_bp.route("/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Request cooperative cancellation of a queued or running job."""
    return jsonify({"status": "success", "job": JobService.cancel(job_id)})


@jobs_bp.route("/<job_id>/retry", methods=["POST"])
def retry_job(job_id):
    """Retry a failed or cancelled job when its handler is still available."""
    return jsonify({"status": "success", "job": JobService.start(job_id)})
