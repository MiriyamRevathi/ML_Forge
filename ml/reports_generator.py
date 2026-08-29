"""
MLForge ML Engine - Platform Report Generator Module
Compiles formatted Markdown and JSON diagnostic reports for Datasets,
Data Quality, EDA, Pipeline Executions, Model Benchmarks, and Drift Reports.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.helpers import get_current_timestamp_readable


class ReportGenerator:
    """
    Automated report compilation engine.
    """

    @staticmethod
    def generate_data_quality_report_md(quality_data: Dict[str, Any]) -> str:
        """
        Generates Markdown report for Data Quality Audit.
        """
        ds_name = quality_data.get("dataset", {}).get("name", "Dataset")
        score = quality_data.get("overall_quality_score", 100)
        checks = quality_data.get("checks", [])

        md = f"# Data Quality Audit Report — {ds_name}\n\n"
        md += f"**Generated At**: {get_current_timestamp_readable()}\n"
        md += f"**Overall Quality Score**: {score}/100\n\n"
        md += "## Quality Rule Execution Summary\n\n"
        md += "| Rule Name | Status | Badge | Details |\n"
        md += "| :--- | :--- | :--- | :--- |\n"

        for c in checks:
            rule = c.get("name", c.get("rule", "Rule"))
            status = c.get("status", "PASS")
            badge = c.get("badge", "")
            msg = c.get("message", "")
            md += f"| {rule} | **{status}** | {badge} | {msg} |\n"

        md += "\n---\n*Report generated automatically by MLForge Data Quality Engine.*\n"
        return md

    @staticmethod
    def generate_pipeline_run_report_md(run_data: Dict[str, Any]) -> str:
        """
        Generates Markdown report for a completed Pipeline Run.
        """
        pipe_name = run_data.get("pipeline_name", "Pipeline")
        run_id = run_data.get("run_id", "")
        duration = run_data.get("duration_seconds", 0)
        metrics = run_data.get("metrics", {})
        logs = run_data.get("logs", [])

        md = f"# Pipeline Execution Report — {pipe_name}\n\n"
        md += f"- **Run ID**: `{run_id}`\n"
        md += f"- **Status**: `{run_data.get('status', 'COMPLETED')}`\n"
        md += f"- **Model Version Generated**: `{run_data.get('model_version', 'N/A')}`\n"
        md += f"- **Execution Duration**: `{duration}s`\n"
        md += f"- **Timestamp**: `{run_data.get('timestamp')}`\n\n"

        md += "## Evaluation Metrics\n\n"
        md += "| Metric Name | Value |\n"
        md += "| :--- | :--- |\n"
        for k, v in metrics.items():
            if k != "confusion_matrix":
                md += f"| {k.replace('_', ' ').title()} | **{v}** |\n"

        md += "\n## Execution Log Output\n\n```text\n"
        for log_line in logs:
            md += f"{log_line}\n"
        md += "```\n\n---\n*MLForge ML Systems Platform Pipeline Report*\n"
        return md

    @staticmethod
    def generate_drift_report_md(drift_data: Dict[str, Any]) -> str:
        """
        Generates Markdown report for Statistical Data Drift Analysis.
        """
        ref_ds = drift_data.get("reference_dataset", "")
        tar_ds = drift_data.get("target_dataset", "")
        status = drift_data.get("drift_status", "NORMAL")
        drifted_count = drift_data.get("drifted_features_count", 0)
        features = drift_data.get("feature_reports", [])

        md = f"# Statistical Data Drift Analysis Report\n\n"
        md += f"- **Reference Dataset**: `{ref_ds}`\n"
        md += f"- **Target Dataset**: `{tar_ds}`\n"
        md += f"- **Overall Drift Status**: **{status}**\n"
        md += f"- **Drifted Features**: **{drifted_count}** / {len(features)} features\n"
        md += f"- **Inspection Time**: `{drift_data.get('timestamp')}`\n\n"

        md += "## Feature-Level Drift Breakdown (KS-Test & Distribution Shifts)\n\n"
        md += "| Feature | Type | Drift Status | p-value | Mean Change % |\n"
        md += "| :--- | :--- | :--- | :--- | :--- |\n"

        for f in features:
            feat = f.get("feature")
            ftype = f.get("type")
            fstatus = f.get("status")
            pval = f.get("p_value", "N/A")
            change = f.get("mean_change_pct", "0.0")
            md += f"| **{feat}** | {ftype} | **{fstatus}** | {pval} | {change}% |\n"

        md += "\n---\n*MLForge Model Monitoring & Drift Detection Engine*\n"
        return md
