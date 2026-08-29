"""
MLForge - Platform Reports Blueprint
Renders platform report summaries, Markdown download views, and report archives.
"""

from flask import Blueprint, render_template, Response, flash, redirect, url_for
from services.dataset_service import DatasetService
from services.pipeline_service import PipelineService
from services.monitoring_service import MonitoringService
from services.reporting_service import ReportingService

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.route("/")
def index():
    """
    Renders Platform Reports archive directory.
    """
    datasets = DatasetService.list_datasets()
    pipeline_runs = PipelineService.list_pipeline_runs()
    drift_reports = MonitoringService.list_drift_reports()

    return render_template(
        "reports/index.html",
        datasets=datasets,
        pipeline_runs=pipeline_runs,
        drift_reports=drift_reports,
        active_tab="reports"
    )


@reports_bp.route("/quality/<dataset_id>")
def quality_report_view(dataset_id):
    """
    Renders formatted Data Quality report.
    """
    md_content = ReportingService.get_quality_report_markdown(dataset_id)
    return render_template(
        "reports/view.html",
        title=f"Quality Audit Report — {dataset_id}",
        markdown_content=md_content,
        active_tab="reports"
    )


@reports_bp.route("/pipeline/<run_id>")
def pipeline_report_view(run_id):
    """
    Renders formatted Pipeline Execution report.
    """
    md_content = ReportingService.get_pipeline_run_markdown(run_id)
    return render_template(
        "reports/view.html",
        title=f"Pipeline Execution Report — {run_id}",
        markdown_content=md_content,
        active_tab="reports"
    )


@reports_bp.route("/drift/<report_id>")
def drift_report_view(report_id):
    """
    Renders formatted Data Drift report.
    """
    md_content = ReportingService.get_drift_report_markdown(report_id)
    return render_template(
        "reports/view.html",
        title=f"Data Drift Report — {report_id}",
        markdown_content=md_content,
        active_tab="reports"
    )


@reports_bp.route("/download/<report_type>/<item_id>")
def download_markdown(report_type, item_id):
    """
    Downloads raw Markdown file for a report.
    """
    if report_type == "quality":
        md = ReportingService.get_quality_report_markdown(item_id)
        filename = f"quality_report_{item_id}.md"
    elif report_type == "pipeline":
        md = ReportingService.get_pipeline_run_markdown(item_id)
        filename = f"pipeline_report_{item_id}.md"
    elif report_type == "drift":
        md = ReportingService.get_drift_report_markdown(item_id)
        filename = f"drift_report_{item_id}.md"
    else:
        md = "# Invalid Report Type"
        filename = "report.md"

    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )
