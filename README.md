# MLForge — Full-Stack Machine Learning Pipeline Platform

**MLForge** is a local, full-stack, web-based Machine Learning Pipeline Builder and ML Systems Management Platform built with Python (Flask, pandas, scikit-learn, joblib, matplotlib) and Vanilla HTML5/CSS3/JavaScript.

MLForge enables end-to-end management of the machine learning lifecycle with **real machine learning execution** on local hardware—no cloud APIs or external databases required.

---

## 🌟 Key Platform Features

- **Visual Pipeline Builder**: Interactive node-based ML pipeline DAG builder with real execution logs.
- **Dataset Ingestion & Quality**: Upload CSVs, preview schema, validate data quality, run exploratory data analysis (EDA) with matplotlib visualization generation.
- **Preprocessing Engine**: Configurable missing value imputation, scaling (Standard, MinMax, Robust), categorical encoding (One-Hot, Ordinal), outlier filtering, and feature interactions.
- **Model Training & Evaluation**: Train Classification (Logistic, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes) and Regression (Linear, Ridge, Lasso, Random Forest, Gradient Boosting) algorithms with real scikit-learn models.
- **Model Registry & Lifecycle**: Version control for trained models, state promotion (`TRAINED` → `VALIDATED` → `STAGING` → `PRODUCTION` → `ARCHIVED`), and rollback support.
- **Prediction System**: Interactive prediction UI with auto-generated feature forms and bulk CSV batch prediction engine.
- **Model Monitoring & Drift Detection**: Real-time statistical drift tracking (KS-test / Kolmogorov-Smirnov, feature mean/std changes, PSI) comparing production data against baseline reference data.
- **Automated Retraining**: Retrain models on incoming batches, evaluate improvements against production baseline, and auto-promote superior model versions.
- **System Diagnostics**: Built-in health check route (`/diagnostics`) inspecting dependencies, file storage systems, and ML pipelines.

---

## 📁 Repository Directory Architecture

```text
mlforge/
├── app.py                      # Flask Application Runner & Factory
├── config.py                   # Platform Environment Configuration & Constants
├── requirements.txt            # Python Dependencies Specification
├── pyproject.toml              # Build & Pytest Configuration
├── Dockerfile                  # Production Docker Container Definition
├── README.md                   # Platform Documentation
│
├── data/                       # Storage Subsystem (Local File-Based DB)
│   ├── datasets/               # Stored CSV Datasets & Metadata JSONs
│   ├── experiments/            # Experiment Run History & Metrics JSONs
│   ├── models/                 # Serialized (.joblib) Models & Version Meta
│   ├── predictions/            # Logged Prediction Artifacts
│   ├── pipelines/              # Saved Pipeline Configurations & Run Logs
│   ├── monitoring/             # Baseline Distributions & Drift Reports
│   ├── sample/                 # Preloaded Benchmark CSV Datasets
│   └── logs/                   # System & Pipeline Log Files
│
├── ml/                         # Machine Learning Core Engine
│   ├── dataset_loader.py       # Dataset Loading & Inspection Engine
│   ├── validation.py           # Automated Quality & Schema Validation
│   ├── exploration.py          # EDA Statistics & Chart Generation Engine
│   ├── cleaning.py             # Imputation, Outlier, and Duplicate Handlers
│   ├── preprocessing.py        # Scalers, Encoders, ColumnTransformers
│   ├── feature_engineering.py  # Feature Transformations & Interaction Generator
│   ├── splitting.py            # Stratified & Train/Test Split Utility
│   ├── training.py             # Sklearn Model Training & Hyperparameter Manager
│   ├── evaluation.py           # Classification & Regression Evaluation Metrics
│   ├── comparison.py           # Multi-Model Evaluation Comparison Engine
│   ├── prediction.py           # Online Single-Prediction Engine
│   ├── batch_prediction.py     # Bulk CSV Prediction Pipeline
│   ├── pipeline.py             # End-to-End Pipeline Execution DAG Controller
│   ├── versioning.py           # Model Versioning & Artifact Manager
│   ├── registry.py             # Model Registry Lifecycle & State Machine
│   ├── monitoring.py           # Health Score & Monitoring Service
│   ├── drift.py                # Statistical Data Drift Analyzer (KS-Test/PSI)
│   └── retraining.py           # Retraining Engine & Auto-Promoter
│
├── services/                   # Business Logic & Service Dispatchers
│   ├── dataset_service.py      # Dataset Metadata & I/O Service
│   ├── experiment_service.py   # Experiment Indexing Service
│   ├── model_service.py        # Model Registry Service
│   ├── pipeline_service.py     # Pipeline Spec & Execution Service
│   ├── monitoring_service.py   # Health Monitoring Service
│   └── prediction_service.py   # Online & Batch Prediction Service
│
├── routes/                     # HTTP Handlers & API Controllers
│   ├── dashboard.py            # Main Dashboard Controller
│   ├── datasets.py             # Dataset Management Routes
│   ├── pipelines.py            # Pipeline Builder & Executor Routes
│   ├── experiments.py          # Experiment Tracking Routes
│   ├── models.py               # Model Registry & Lifecycle Routes
│   ├── predictions.py          # Prediction & Batch Execution Routes
│   ├── monitoring.py           # Health & Drift Analytics Routes
│   └── diagnostics.py          # Platform Health Diagnostic Route
│
├── utils/                      # Shared System Utilities
│   ├── files.py                # Path Security & Safe File I/O Helpers
│   ├── logging.py              # Central Logger & Pipeline Execution Buffer
│   ├── metrics.py              # Math & Sklearn Metrics Aggregator
│   ├── validation.py           # Request Payload & File Schema Validators
│   └── helpers.py              # Formatting & Date Helper Functions
│
├── templates/                  # Server-Rendered HTML Components
│   ├── base.html               # Main Navigation & Theme Container
│   ├── dashboard.html          # Key Systems Overview Dashboard
│   ├── diagnostics.html        # Platform Diagnostics Dashboard
│   ├── datasets/               # Dataset List, View, Upload, EDA UI
│   ├── pipelines/              # Visual Builder & Execution History UI
│   ├── experiments/            # Experiment Run History & Metrics UI
│   ├── models/                 # Model Registry UI
│   ├── predictions/            # Prediction Form & CSV Batch UI
│   └── monitoring/             # Monitoring, Drift & Retraining UI
│
├── static/                     # User Interface Assets
│   ├── css/                    # Modular Responsive CSS Architecture
│   └── js/                     # Modular Vanilla JavaScript Application Logic
│
└── tests/                      # Pytest Test Suite
```

---

## 🚀 Quickstart & Setup Guide

### 1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Sample Datasets & Launch Platform

```bash
# Generate sample benchmark datasets (Customer Churn, House Prices, Iris)
python data/sample/generate_samples.py

# Launch Flask Web Server
python app.py
```

Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 🧪 Testing

Execute the test suite with `pytest`:

```bash
pytest
```

---

## 🐳 Running with Docker

```bash
docker build -t mlforge:latest .
docker run -p 5000:5000 mlforge:latest
```

---

## 🔒 Security & Data Local Isolation

- All model training, prediction, data validation, and drift analysis are performed locally inside the Python process.
- No remote calls, external APIs, cloud compute resources, or third-party databases are utilized.
- File uploads are validated with strict extensions (`.csv`, `.json`), path-traversal safeguards, and isolated file naming conventions.
#   M L _ F o r g e  
 