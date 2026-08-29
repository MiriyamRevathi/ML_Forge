"""
MLForge - Utility Functions Unit Tests
Tests file security, metrics calculation, validation checks, and helper functions.
"""

import pytest
import numpy as np
from utils.files import sanitize_filename, validate_path_safety, FileSystemError
from utils.metrics import calculate_classification_metrics, calculate_regression_metrics
from utils.validation import validate_pipeline_configuration, ValidationError
from utils.helpers import generate_unique_id, format_duration


def test_sanitize_filename():
    """Tests filename sanitization."""
    assert sanitize_filename("my_dataset.csv") == "my_dataset.csv"
    assert sanitize_filename("../../../etc/passwd.csv") == "etc_passwd.csv"


def test_path_safety_traversal_detection(tmp_path):
    """Tests path traversal detection."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")
    
    with pytest.raises(FileSystemError):
        validate_path_safety(outside_file, base_dir)


def test_classification_metrics_computation():
    """Tests accuracy, precision, recall, and f1 computation."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == 0.75


def test_regression_metrics_computation():
    """Tests MAE, MSE, RMSE, R2 computation."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 305.0, 395.0])
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert metrics["r2_score"] > 0.95


def test_validate_pipeline_configuration():
    """Tests valid and invalid pipeline configuration checks."""
    valid_cfg = {
        "name": "Test Pipeline",
        "dataset_id": "customer_churn.csv",
        "target_column": "churn",
        "task": "classification",
        "test_size": 0.2
    }
    assert validate_pipeline_configuration(valid_cfg) is True
    
    invalid_cfg = {
        "name": "Test Pipeline",
        "dataset_id": "",
        "target_column": "churn",
        "task": "invalid_task"
    }
    with pytest.raises(ValidationError):
        validate_pipeline_configuration(invalid_cfg)


def test_generate_unique_id():
    """Tests unique ID generator."""
    uid = generate_unique_id("ds")
    assert uid.startswith("ds_")
    assert len(uid) > 10


def test_format_duration():
    """Tests duration formatting."""
    assert "ms" in format_duration(0.45)
    assert "s" in format_duration(12.34)
    assert "m" in format_duration(125.0)
