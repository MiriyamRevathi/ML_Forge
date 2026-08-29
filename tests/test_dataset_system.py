"""
MLForge - Phase 2 Dataset System Unit Tests
Tests dataset schema inspection, quality validation suite, and EDA chart rendering.
"""

import pytest
import pandas as pd
import numpy as np
from app import create_app
from ml.dataset_loader import DatasetLoader
from ml.validation import DataValidator
from ml.exploration import ExploratoryDataAnalysis
from services.dataset_service import DatasetService


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, np.nan],
        "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000],
        "category": ["A", "B", "A", "B", "C", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0]
    })


def test_schema_inspection(sample_df):
    """Tests schema inspection output."""
    schema = DatasetLoader.inspect_schema(sample_df)
    assert schema["total_rows"] == 7
    assert schema["total_columns"] == 4
    assert "age" in schema["numerical_columns"]
    assert "category" in schema["categorical_columns"]


def test_data_validation(sample_df):
    """Tests automated validation checks."""
    report = DataValidator.validate_dataset(sample_df, target_column="target")
    assert report["is_valid"] is True
    assert report["score"] > 50
    assert len(report["checks"]) >= 6


def test_eda_statistics_computation(sample_df):
    """Tests EDA numerical and categorical statistics functions."""
    num_stats = ExploratoryDataAnalysis.get_numerical_statistics(sample_df)
    assert "income" in num_stats
    assert num_stats["income"]["mean"] == 80000.0
    
    cat_stats = ExploratoryDataAnalysis.get_categorical_statistics(sample_df)
    assert "category" in cat_stats
    assert cat_stats["category"]["unique"] == 3


def test_eda_chart_generation(sample_df):
    """Tests Matplotlib base64 chart rendering."""
    dist_chart = ExploratoryDataAnalysis.create_distribution_chart(sample_df)
    assert dist_chart is not None
    assert dist_chart.startswith("data:image/png;base64,")

    boxplot_chart = ExploratoryDataAnalysis.create_box_plot_chart(sample_df)
    assert boxplot_chart is not None
    assert boxplot_chart.startswith("data:image/png;base64,")

    corr_chart = ExploratoryDataAnalysis.create_correlation_heatmap(sample_df)
    assert corr_chart is not None
    assert corr_chart.startswith("data:image/png;base64,")


def test_dataset_routes_validation_and_eda():
    """Verifies validation and EDA routes on Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        # Get first dataset ID
        datasets = DatasetService.list_datasets()
        if datasets:
            ds_id = datasets[0]["id"]
            
            val_res = client.get(f"/datasets/{ds_id}/validation")
            assert val_res.status_code == 200
            assert b"Quality Score" in val_res.data

            eda_res = client.get(f"/datasets/{ds_id}/eda")
            assert eda_res.status_code == 200
            assert b"Exploratory Data Analysis" in eda_res.data
