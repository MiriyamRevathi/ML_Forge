"""
MLForge - Services Layer Unit Tests
Tests dataset service, pipeline service, experiment service, model service, and monitoring service.
"""

import pytest
from services.dataset_service import DatasetService
from services.pipeline_service import PipelineService
from services.experiment_service import ExperimentService
from services.model_service import ModelService
from services.monitoring_service import MonitoringService


def test_dataset_service_list():
    """Verifies that sample datasets are registered and listed."""
    datasets = DatasetService.list_datasets()
    assert len(datasets) >= 3
    dataset_ids = [d["id"] for d in datasets]
    assert "customer_churn" in dataset_ids or "customer_churn_meta" in dataset_ids or any("customer_churn" in d_id for d_id in dataset_ids)


def test_pipeline_service_save_and_get():
    """Verifies pipeline configuration persistence."""
    cfg = {
        "name": "Unit Test Pipeline",
        "dataset_id": "customer_churn.csv",
        "target_column": "churn",
        "task": "classification"
    }
    saved = PipelineService.save_pipeline(cfg)
    assert "id" in saved
    
    retrieved = PipelineService.get_pipeline(saved["id"])
    assert retrieved is not None
    assert retrieved["name"] == "Unit Test Pipeline"


def test_experiment_service_log_and_list():
    """Verifies experiment logging and listing."""
    exp = ExperimentService.log_experiment(
        name="Test Experiment Run",
        dataset_name="customer_churn.csv",
        target="churn",
        task="classification",
        model_name="Random Forest",
        hyperparameters={"n_estimators": 100},
        metrics={"accuracy": 0.92, "accuracy_percentage": 92.0},
        training_duration_seconds=1.45
    )
    assert exp["id"].startswith("exp_")
    
    experiments = ExperimentService.list_experiments()
    assert len(experiments) >= 1


def test_monitoring_service_dashboard_summary():
    """Verifies dashboard summary calculations."""
    summary = MonitoringService.get_dashboard_summary()
    assert "datasets_count" in summary
    assert "health_score" in summary
    assert summary["health_score"] >= 0
