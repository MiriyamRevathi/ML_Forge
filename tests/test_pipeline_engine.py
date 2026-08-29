"""
MLForge - Phase 3 & 4 Pipeline Execution Engine Unit Tests
Tests end-to-end execution of Classification and Regression ML Pipeline DAGs.
"""

import pytest
from ml.pipeline import PipelineEngine
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.experiment_service import ExperimentService


def test_classification_pipeline_execution():
    """Executes full Classification ML Pipeline on Customer Churn dataset."""
    datasets = DatasetService.list_datasets()
    churn_ds = next((d for d in datasets if "churn" in d["id"]), datasets[0])
    
    config = {
        "name": "Customer Churn Unit Test Pipeline",
        "dataset_id": churn_ds["id"],
        "target_column": churn_ds["target_column"],
        "task": "classification",
        "test_size": 0.2,
        "preprocessing": {
            "impute_strategy": "mean",
            "scaler": "standard",
            "encoder": "onehot"
        },
        "model": {
            "name": "random_forest",
            "hyperparameters": {
                "n_estimators": 50,
                "max_depth": 5
            }
        }
    }
    
    engine = PipelineEngine(config)
    run_result = engine.execute()
    
    assert run_result["status"] == "COMPLETED"
    assert "model_version" in run_result
    assert "metrics" in run_result
    assert run_result["metrics"]["accuracy"] > 0.5
    assert len(run_result["logs"]) >= 8
    
    # Check that model version was saved to registry
    model_meta = ModelService.get_model_metadata(run_result["model_version"])
    assert model_meta is not None
    assert model_meta["task_type"] == "classification"


def test_regression_pipeline_execution():
    """Executes full Regression ML Pipeline on House Prices dataset."""
    datasets = DatasetService.list_datasets()
    house_ds = next((d for d in datasets if "house" in d["id"]), datasets[0])
    
    config = {
        "name": "House Prices Unit Test Pipeline",
        "dataset_id": house_ds["id"],
        "target_column": house_ds["target_column"],
        "task": "regression",
        "test_size": 0.2,
        "preprocessing": {
            "impute_strategy": "median",
            "scaler": "robust",
            "encoder": "onehot"
        },
        "model": {
            "name": "random_forest_regressor",
            "hyperparameters": {
                "n_estimators": 50
            }
        }
    }
    
    engine = PipelineEngine(config)
    run_result = engine.execute()
    
    assert run_result["status"] == "COMPLETED"
    assert "model_version" in run_result
    assert "metrics" in run_result
    assert "r2_score" in run_result["metrics"]
    assert run_result["metrics"]["r2_score"] > 0.4
    
    # Verify experiment was logged
    experiments = ExperimentService.list_experiments()
    assert len(experiments) >= 1
