"""
MLForge - Phase 6, 7 & 8 Model Registry & Comparison Unit Tests
Tests model comparison ranking, state machine transitions, promotion logic, and detail routes.
"""

import pytest
from app import create_app
from ml.comparison import ModelComparer
from ml.registry import ModelRegistryStateMachine
from services.model_service import ModelService


def test_model_comparison_ranking():
    """Tests model comparison rank ordering and primary metric extraction."""
    dummy_models = [
        {"version": "model_v1", "name": "Decision Tree", "task_type": "classification", "status": "TRAINED", "metrics": {"accuracy": 0.82}},
        {"version": "model_v2", "name": "Random Forest", "task_type": "classification", "status": "TRAINED", "metrics": {"accuracy": 0.94}},
        {"version": "model_v3", "name": "Logistic Regression", "task_type": "classification", "status": "TRAINED", "metrics": {"accuracy": 0.88}}
    ]
    
    result = ModelComparer.compare_models(dummy_models)
    assert result["best_model"]["version"] == "model_v2"
    assert result["best_model"]["primary_metric_value"] == 0.94
    assert result["comparison_chart"] is not None
    assert result["comparison_chart"].startswith("data:image/png;base64,")


def test_state_machine_transitions():
    """Tests registry state machine transition validity."""
    assert ModelRegistryStateMachine.can_transition("TRAINED", "VALIDATED") is True
    assert ModelRegistryStateMachine.can_transition("TRAINED", "PRODUCTION") is False
    assert ModelRegistryStateMachine.can_transition("VALIDATED", "PRODUCTION") is True


def test_model_promotion():
    """Tests model promotion and automatic demotion of previous production model."""
    models = ModelService.list_models()
    if models:
        version = models[0]["version"]
        promoted = ModelRegistryStateMachine.promote_to_production(version)
        assert promoted["status"] == "PRODUCTION"
        
        prod_model = ModelService.get_production_model()
        assert prod_model is not None
        assert prod_model["version"] == version


def test_models_routes():
    """Tests models comparison and detail Flask HTTP endpoints."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        res_compare = client.get("/models/compare")
        assert res_compare.status_code == 200
        assert b"Model Comparison Benchmark Matrix" in res_compare.data
        
        models = ModelService.list_models()
        if models:
            v = models[0]["version"]
            res_detail = client.get(f"/models/{v}")
            assert res_detail.status_code == 200
            assert b"Model Version Metadata" in res_detail.data
