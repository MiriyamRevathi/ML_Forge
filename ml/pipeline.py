"""
MLForge ML Engine - End-to-End Pipeline Execution Engine
Orchestrates real machine-learning pipeline DAG execution:
Load -> Validate -> Clean -> Preprocess -> Engineer Features -> Split -> Train -> Evaluate -> Save Version & Log Experiment.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logging import PipelineExecutionLogger
from utils.helpers import generate_unique_id, get_current_timestamp_iso
from services.dataset_service import DatasetService
from services.experiment_service import ExperimentService
from services.pipeline_service import PipelineService
from ml.validation import DataValidator
from ml.cleaning import DataCleaner
from ml.preprocessing import FeaturePreprocessor
from ml.feature_engineering import FeatureEngineer
from ml.splitting import DatasetSplitter
from ml.training import ModelTrainer
from ml.evaluation import ModelEvaluator
from ml.versioning import ModelVersionManager


class PipelineEngine:
    """
    Core executor for ML DAG pipelines.
    """
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        self.config = pipeline_config
        self.run_id = generate_unique_id("run")
        self.logger = PipelineExecutionLogger(self.run_id)

    def execute(self) -> Dict[str, Any]:
        """
        Executes the complete end-to-end ML Pipeline.
        """
        start_time = time.time()
        self.logger.info(f"Starting pipeline execution for: '{self.config.get('name', 'Pipeline')}'")
        
        dataset_id = self.config.get("dataset_id")
        target_col = self.config.get("target_column")
        task_type = self.config.get("task", "classification")
        test_size = float(self.config.get("test_size", 0.2))
        
        prep_config = self.config.get("preprocessing", {})
        impute_strat = prep_config.get("impute_strategy", "mean")
        scaler_choice = prep_config.get("scaler", "standard")
        encoder_choice = prep_config.get("encoder", "onehot")
        
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "random_forest")
        hyperparams = model_config.get("hyperparameters", {})

        # Step 1: Load Dataset
        self.logger.info(f"Step 1: Loading dataset '{dataset_id}'...")
        df_raw = DatasetService.load_dataset_dataframe(dataset_id)
        self.logger.info(f"Dataset loaded successfully: {len(df_raw)} rows, {len(df_raw.columns)} columns.")

        # Step 2: Validate Data
        self.logger.info("Step 2: Executing data quality validation...")
        val_report = DataValidator.validate_dataset(df_raw, target_column=target_col)
        self.logger.info(f"Validation completed. Quality score: {val_report['score']}/100.")

        # Step 3: Clean Data
        self.logger.info("Step 3: Cleaning dataset (handling missing values & duplicates)...")
        df_clean, dup_count = DataCleaner.remove_duplicates(df_raw)
        if dup_count > 0:
            self.logger.info(f"Removed {dup_count} duplicate rows.")
            
        df_clean, missing_info = DataCleaner.handle_missing_values(
            df_clean,
            strategy=impute_strat,
            target_column=target_col
        )
        self.logger.info(f"Data cleaning completed. Remaining missing cells: {missing_info['remaining_missing']}.")

        # Ensure target column exists
        if target_col not in df_clean.columns:
            target_col = df_clean.columns[-1]
            self.logger.warning(f"Target column adjusted to: '{target_col}'")

        # Step 4: Preprocessing & Scaling
        self.logger.info(f"Step 4: Preprocessing features (Scaler: {scaler_choice}, Encoder: {encoder_choice})...")
        X_raw = df_clean.drop(columns=[target_col])
        y_raw = df_clean[target_col]

        # Apply interaction features if numerical columns exist
        X_raw = FeatureEngineer.apply_interaction_features(X_raw, max_features=3)

        preprocessor = FeaturePreprocessor(
            scaler_type=scaler_choice,
            encoder_type=encoder_choice,
            impute_strategy=impute_strat
        )
        X_processed, feature_names = preprocessor.fit_transform(X_raw)
        self.logger.info(f"Feature matrix preprocessed: {X_processed.shape[1]} features output.")

        # Re-assemble processed dataframe for split
        df_processed = pd.DataFrame(X_processed, columns=feature_names)
        df_processed[target_col] = y_raw.values

        # Step 5: Split Dataset
        self.logger.info(f"Step 5: Performing Train/Test split (test_size={test_size})...")
        X_train, X_test, y_train, y_test = DatasetSplitter.split_dataset(
            df_processed,
            target_column=target_col,
            test_size=test_size,
            task_type=task_type
        )
        self.logger.info(f"Split complete: Train set {X_train.shape[0]} samples, Test set {X_test.shape[0]} samples.")

        # Step 6: Model Training
        self.logger.info(f"Step 6: Training ML Model ({model_name})...")
        model = ModelTrainer.train_model(
            task_type=task_type,
            model_name=model_name,
            X_train=X_train.values,
            y_train=y_train.values,
            hyperparameters=hyperparams
        )
        self.logger.info(f"Model training completed successfully.")

        # Step 7: Model Evaluation
        self.logger.info("Step 7: Evaluating model performance on test set...")
        metrics, y_pred, y_prob = ModelEvaluator.evaluate(
            model=model,
            task_type=task_type,
            X_test=X_test.values,
            y_test=y_test.values
        )
        
        primary_metric_str = f"Accuracy: {metrics.get('accuracy_percentage')}%" if task_type == "classification" else f"R²: {metrics.get('r2_score')}"
        self.logger.info(f"Evaluation completed. Performance: {primary_metric_str}")

        duration = round(time.time() - start_time, 2)
        self.logger.info(f"Pipeline execution finished in {duration}s.")

        # Step 8: Version Model & Track Experiment
        model_version_meta = ModelVersionManager.create_version(
            model_object=model,
            model_name=f"{model_name.replace('_', ' ').title()} ({self.config.get('name')})",
            task_type=task_type,
            dataset_name=str(dataset_id),
            target_column=target_col,
            feature_names=feature_names,
            metrics=metrics,
            hyperparameters=hyperparams,
            preprocessor_object=preprocessor,
            pipeline_id=self.config.get("id")
        )
        
        exp_record = ExperimentService.log_experiment(
            name=f"Run - {self.config.get('name')}",
            dataset_name=str(dataset_id),
            target=target_col,
            task=task_type,
            model_name=model_name,
            hyperparameters=hyperparams,
            metrics=metrics,
            training_duration_seconds=duration,
            model_version=model_version_meta["version"],
            pipeline_id=self.config.get("id")
        )

        run_summary = {
            "run_id": self.run_id,
            "pipeline_id": self.config.get("id"),
            "pipeline_name": self.config.get("name"),
            "status": "COMPLETED",
            "duration_seconds": duration,
            "metrics": metrics,
            "model_version": model_version_meta["version"],
            "experiment_id": exp_record["id"],
            "logs": self.logger.get_formatted_logs(),
            "timestamp": get_current_timestamp_iso()
        }

        PipelineService.save_pipeline_run(run_summary)
        return run_summary

# Feature sync: feature/pipeline-dag-engine (PR #6)

# Feature sync: feature/pipeline-dag-engine (PR #6)

# Feature sync: feature/pipeline-dag-engine (PR #6)
