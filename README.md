# MLForge — Machine Learning Pipeline Platform

MLForge is a local full-stack web application for building, training, evaluating, and managing machine learning pipelines.

It is built with **Python, Flask, pandas, scikit-learn, joblib, matplotlib, HTML, CSS, and Vanilla JavaScript**. Everything runs locally without cloud APIs or an external database.

## Features

* **Pipeline Builder** — Create and execute end-to-end ML pipelines.
* **Dataset Management** — Upload CSV datasets, preview data, validate schemas, and inspect quality.
* **EDA** — Generate statistics, distributions, correlations, and visualizations.
* **Data Preprocessing** — Handle missing values, scaling, encoding, outliers, and feature transformations.
* **Model Training** — Train classification and regression models using scikit-learn.
* **Model Evaluation** — Calculate accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R², and other metrics.
* **Experiment Tracking** — Store training parameters, metrics, datasets, and experiment results.
* **Model Registry** — Manage model versions and lifecycle states.
* **Prediction** — Perform single and batch CSV predictions.
* **Model Monitoring** — Track model and prediction behavior.
* **Drift Detection** — Detect feature distribution changes using statistical methods such as KS-test and PSI.
* **Retraining** — Train new model versions and compare them with existing models.
* **Diagnostics** — Check application dependencies and ML subsystem health.

## Technology Stack

| Layer            | Technology                      |
| ---------------- | ------------------------------- |
| Backend          | Python, Flask                   |
| Machine Learning | scikit-learn                    |
| Data Processing  | pandas, NumPy                   |
| Visualization    | matplotlib                      |
| Model Storage    | joblib                          |
| Frontend         | HTML5, CSS3, Vanilla JavaScript |
| Storage          | Local files / JSON              |
| Testing          | pytest                          |
| Deployment       | Docker                          |

## Project Structure

```text
mlforge/
├── app.py
├── config.py
├── requirements.txt
├── pyproject.toml
├── Dockerfile
│
├── ml/
│   ├── dataset_loader.py
│   ├── validation.py
│   ├── exploration.py
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── splitting.py
│   ├── training.py
│   ├── evaluation.py
│   ├── comparison.py
│   ├── prediction.py
│   ├── batch_prediction.py
│   ├── pipeline.py
│   ├── registry.py
│   ├── monitoring.py
│   ├── drift.py
│   └── retraining.py
│
├── services/
├── routes/
├── utils/
├── templates/
├── static/
├── data/
└── tests/
```

## Installation

### 1. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate sample datasets

```bash
python data/sample/generate_samples.py
```

### 4. Start MLForge

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Machine Learning Workflow

MLForge supports the following workflow:

```text
Dataset
   ↓
Validation
   ↓
Data Cleaning
   ↓
EDA
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Experiment Tracking
   ↓
Model Registry
   ↓
Prediction
   ↓
Monitoring
   ↓
Drift Detection
   ↓
Retraining
```

## Supported Models

### Classification

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine
* K-Nearest Neighbors
* Naive Bayes
* Decision Tree
* Extra Trees

### Regression

* Linear Regression
* Ridge
* Lasso
* ElasticNet
* Random Forest
* Gradient Boosting
* Decision Tree
* Extra Trees

## Testing

Run the test suite with:

```bash
pytest
```

## Docker

Build the image:

```bash
docker build -t mlforge:latest .
```

Run the application:

```bash
docker run -p 5000:5000 mlforge:latest
```

Then open:

```text
http://127.0.0.1:5000
```

## Local-First Architecture

MLForge is designed to operate locally.

* No external database is required.
* No cloud ML service is required.
* Models are stored locally.
* Dataset and experiment metadata are stored locally.
* Training and prediction run on local hardware.
* Drift analysis and monitoring run locally.

## Project Goal

MLForge demonstrates the complete **Machine Learning Systems lifecycle**, from dataset ingestion and preprocessing to model training, deployment-style prediction, monitoring, drift detection, and retraining.
