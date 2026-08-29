"""
MLForge - Core Configuration Module
Defines system-wide paths, runtime parameters, data storage limits, and task constants.
"""

import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Data Storage Directories
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "datasets"
EXPERIMENT_DIR = DATA_DIR / "experiments"
MODEL_DIR = DATA_DIR / "models"
PREDICTION_DIR = DATA_DIR / "predictions"
PIPELINE_DIR = DATA_DIR / "pipelines"
MONITORING_DIR = DATA_DIR / "monitoring"
SAMPLE_DIR = DATA_DIR / "sample"
LOG_DIR = DATA_DIR / "logs"

# Ensure all storage directories exist at import time
for storage_path in [
    DATA_DIR,
    DATASET_DIR,
    EXPERIMENT_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    PIPELINE_DIR,
    MONITORING_DIR,
    SAMPLE_DIR,
    LOG_DIR,
]:
    storage_path.mkdir(parents=True, exist_ok=True)

# Application Parameters
SECRET_KEY = os.environ.get("MLFORGE_SECRET_KEY", "mlforge-secret-key-production-local-998822")
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload limit
ALLOWED_EXTENSIONS = {"csv", "json"}

# ML Task Definitions
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
SUPPORTED_TASKS = [TASK_CLASSIFICATION, TASK_REGRESSION]

# Supported ML Models Architecture Catalogue
CLASSIFICATION_MODELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree Classifier",
    "random_forest": "Random Forest Classifier",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Classifier (SVC)",
    "naive_bayes": "Gaussian Naive Bayes",
    "gradient_boosting": "Gradient Boosting Classifier",
}

REGRESSION_MODELS = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "decision_tree_regressor": "Decision Tree Regressor",
    "random_forest_regressor": "Random Forest Regressor",
    "gradient_boosting_regressor": "Gradient Boosting Regressor",
}

# Model Lifecycle Statuses
STATUS_TRAINED = "TRAINED"
STATUS_VALIDATED = "VALIDATED"
STATUS_STAGING = "STAGING"
STATUS_PRODUCTION = "PRODUCTION"
STATUS_ARCHIVED = "ARCHIVED"

REGISTRY_STATUSES = [
    STATUS_TRAINED,
    STATUS_VALIDATED,
    STATUS_STAGING,
    STATUS_PRODUCTION,
    STATUS_ARCHIVED,
]

# Health & Monitoring Thresholds
HEALTH_EXCELLENT_THRESHOLD = 90
HEALTH_GOOD_THRESHOLD = 75
HEALTH_WARNING_THRESHOLD = 60

# Version String Prefix
MODEL_VERSION_PREFIX = "v"
