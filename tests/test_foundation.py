"""
MLForge - Foundation & Flask Integration Unit Tests
Verifies app initialization, route registration, 200 OK HTTP responses, and diagnostics.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_dashboard_route_status_200(client):
    """Verifies that the dashboard home route loads with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"MLForge" in response.data


def test_datasets_route_status_200(client):
    """Verifies datasets route response."""
    response = client.get("/datasets/")
    assert response.status_code == 200


def test_pipelines_route_status_200(client):
    """Verifies pipelines route response."""
    response = client.get("/pipelines/")
    assert response.status_code == 200


def test_experiments_route_status_200(client):
    """Verifies experiments route response."""
    response = client.get("/experiments/")
    assert response.status_code == 200


def test_models_route_status_200(client):
    """Verifies model registry route response."""
    response = client.get("/models/")
    assert response.status_code == 200


def test_predictions_route_status_200(client):
    """Verifies predictions route response."""
    response = client.get("/predictions/")
    assert response.status_code == 200


def test_monitoring_route_status_200(client):
    """Verifies monitoring route response."""
    response = client.get("/monitoring/")
    assert response.status_code == 200


def test_diagnostics_route_status_200(client):
    """Verifies system diagnostics route response."""
    response = client.get("/diagnostics/")
    assert response.status_code == 200
    assert b"ALL SYSTEMS OPERATIONAL" in response.data or b"SYSTEM ISSUES DETECTED" in response.data


def test_diagnostics_api_status_200(client):
    """Verifies diagnostics JSON API response."""
    response = client.get("/diagnostics/api/status")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert "diagnostics" in json_data
