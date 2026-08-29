"""
MLForge - Phase 9, 10, 11 & 12 Unit Tests
Tests online single inference, bulk CSV batch predictions, Kolmogorov-Smirnov drift detection,
and automated model retraining pipelines.
"""

import pytest
import pandas as pd
from app import create_app
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.pipeline_service import PipelineService
from ml.prediction import SinglePredictor
from ml.batch_prediction import BatchPredictor
from ml.drift import DataDriftDetector
from ml.retraining import ModelRetrainer


@pytest.fixture
def active_model_version():
    models = ModelService.list_models()
    if models:
        return models[0]["version"]
    pytest.skip("No trained models available for testing.")


def test_single_prediction(active_model_version):
    """Tests single inference execution on trained model."""
    meta = ModelService.get_model_metadata(active_model_version)
    bundle = ModelService.load_model_artifact(active_model_version)
    preprocessor = bundle.get("preprocessor")
    
    if preprocessor:
        raw_num = getattr(preprocessor, "numerical_features", [])
        raw_cat = getattr(preprocessor, "categorical_features", [])
        input_data = {col: 1.0 if col in raw_num else "A" for col in (raw_num + raw_cat)}
    else:
        features = meta.get("feature_names", [])
        input_data = {feat: 1.0 for feat in features}
    
    result = SinglePredictor.predict(active_model_version, input_data)
    assert "prediction" in result
    assert result["model_version"] == active_model_version


def test_batch_csv_prediction(tmp_path, active_model_version):
    """Tests bulk CSV batch prediction engine."""
    meta = ModelService.get_model_metadata(active_model_version)
    bundle = ModelService.load_model_artifact(active_model_version)
    preprocessor = bundle.get("preprocessor")
    
    if preprocessor:
        raw_num = getattr(preprocessor, "numerical_features", [])
        raw_cat = getattr(preprocessor, "categorical_features", [])
        input_data = {col: 1.0 if col in raw_num else "A" for col in (raw_num + raw_cat)}
    else:
        features = meta.get("feature_names", [])
        input_data = {feat: 1.0 for feat in features}
    
    dummy_df = pd.DataFrame([input_data for _ in range(10)])
    csv_file = tmp_path / "test_batch.csv"
    dummy_df.to_csv(csv_file, index=False)
    
    batch_res = BatchPredictor.process_batch_csv(active_model_version, csv_file, "test_batch.csv")
    assert batch_res["total_rows"] == 10
    assert batch_res["processed_rows"] == 10
    assert "output_filename" in batch_res


def test_data_drift_detection():
    """Tests Kolmogorov-Smirnov statistical data drift analysis."""
    datasets = DatasetService.list_datasets()
    if datasets:
        ds1 = datasets[0]["id"]
        
        report = DataDriftDetector.detect_drift(ds1, ds1)
        assert "has_drift" in report
        assert "drift_status" in report
        assert "feature_reports" in report
        assert len(report["feature_reports"]) >= 1


def test_automated_model_retraining():
    """Tests pipeline retraining and auto-promotion evaluation."""
    pipelines = PipelineService.list_pipelines()
    if pipelines:
        pipe_id = pipelines[0]["id"]
        retrain_res = ModelRetrainer.retrain_model(pipe_id)
        assert "candidate_version" in retrain_res
        assert "promoted_to_production" in retrain_res
        assert "promotion_reason" in retrain_res


def test_prediction_and_monitoring_routes():
    """Tests Flask HTTP routes for prediction schema, drift, and monitoring."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        models = ModelService.list_models()
        if models:
            v = models[0]["version"]
            res_schema = client.get(f"/predictions/api/schema/{v}")
            assert res_schema.status_code == 200
            assert res_schema.get_json()["status"] == "success"

        res_mon = client.get("/monitoring/")
        assert res_mon.status_code == 200
        assert b"Model Performance & Health Score Overview" in res_mon.data
