"""
MLForge ML Engine - Pipeline Execution Orchestrator Module
Executes pipeline DAG stage nodes sequentially, manages context state propagation,
writes execution logs to disk, saves artifacts, and records run metrics.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from utils.logging import PipelineExecutionLogger
from utils.helpers import generate_unique_id, get_current_timestamp_iso
from services.dataset_service import DatasetService
from services.pipeline_service import PipelineService
from services.experiment_service import ExperimentService
from services.model_service import ModelService
from ml.cleaning import DataCleaner
from ml.preprocessing import PreprocessorBuilder
from ml.feature_engineering import FeatureEngineer
from ml.splitting import DataSplitter
from ml.training import ModelTrainer
from ml.evaluation import ModelEvaluator


class PipelineExecutor:
    """
    DAG Pipeline Execution Engine.
    """

    @staticmethod
    def run_pipeline(pipeline_id: str) -> Dict[str, Any]:
        """
        Executes a saved DAG pipeline specification end-to-end.
        """
        start_time = time.time()
        run_id = generate_unique_id("run")

        pipe_spec = PipelineService.get_pipeline(pipeline_id)
        if not pipe_spec:
            raise FileNotFoundError(f"Pipeline ID '{pipeline_id}' not found.")

        pipe_name = pipe_spec.get("name", "Pipeline")
        dataset_id = pipe_spec.get("dataset_id")
        target_col = pipe_spec.get("target_column")
        task_type = pipe_spec.get("task_type", "classification")

        logger = PipelineExecutionLogger(run_id=run_id, pipeline_name=pipe_name)
        logger.info(f"Starting execution of pipeline '{pipe_name}' (ID: {pipeline_id})")

        # Step 1: Load Dataset
        logger.info(f"Stage 1 [Ingestion]: Loading dataset '{dataset_id}'")
        df = DatasetService.load_dataset_dataframe(dataset_id)
        meta = DatasetService.get_dataset_metadata(dataset_id)
        logger.info(f"Loaded DataFrame with shape {df.shape}")

        if not target_col or target_col not in df.columns:
            target_col = meta.get("target_column") or df.columns[-1]

        # Step 2: Data Cleaning
        cleaning_config = pipe_spec.get("cleaning", {})
        logger.info("Stage 2 [Cleaning]: Executing missing value imputation and outlier dropping")
        df_clean = DataCleaner.clean_dataset(
            df=df,
            drop_duplicates=cleaning_config.get("drop_duplicates", True),
            missing_strategy=cleaning_config.get("missing_strategy", "mean"),
            outlier_method=cleaning_config.get("outlier_method", "none")
        )
        logger.info(f"Cleaned DataFrame shape: {df_clean.shape}")

        # Step 3: Feature Engineering
        feat_config = pipe_spec.get("feature_engineering", {})
        logger.info("Stage 3 [Feature Engineering]: Applying mathematical transforms & interactions")
        df_feat = FeatureEngineer.apply_feature_engineering(
            df=df_clean,
            target_column=target_col,
            enable_interactions=feat_config.get("enable_interactions", False),
            enable_log_transforms=feat_config.get("enable_log_transforms", False)
        )
        logger.info(f"Engineered DataFrame shape: {df_feat.shape}")

        # Step 4: Split Train/Test
        split_ratio = float(pipe_spec.get("train_test_split", 0.8))
        logger.info(f"Stage 4 [Splitting]: Splitting dataset with train ratio {split_ratio}")
        X_train, X_test, y_train, y_test, feature_names = DataSplitter.split_dataset(
            df=df_feat,
            target_column=target_col,
            train_ratio=split_ratio,
            stratify=(task_type == "classification")
        )
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Step 5: Preprocessing & Scaling
        prep_config = pipe_spec.get("preprocessing", {})
        logger.info("Stage 5 [Preprocessing]: Fitting Scalers & Encoders ColumnTransformer")
        preprocessor = PreprocessorBuilder.build_preprocessor(
            df=df_feat,
            target_column=target_col,
            scaling_method=prep_config.get("scaling", "standard"),
            encoding_method=prep_config.get("encoding", "onehot")
        )
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        logger.info(f"Processed feature matrix shape: {X_train_proc.shape}")

        # Step 6: Model Training
        model_config = pipe_spec.get("model", {})
        model_name = model_config.get("algorithm", "random_forest")
        params = model_config.get("hyperparameters", {})

        logger.info(f"Stage 6 [Training]: Fitting model '{model_name}' on task '{task_type}'")
        model = ModelTrainer.train_model(
            task_type=task_type,
            algorithm=model_name,
            X_train=X_train_proc,
            y_train=y_train,
            hyperparameters=params
        )

        # Step 7: Model Evaluation
        logger.info("Stage 7 [Evaluation]: Calculating test performance metrics")
        evaluator = ModelEvaluator(model=model, task_type=task_type)
        eval_results = evaluator.evaluate(X_test=X_test_proc, y_test=y_test)
        logger.info(f"Evaluation Metrics: {eval_results['metrics']}")

        # Step 8: Version Model & Register Artifact
        duration = round(time.time() - start_time, 3)
        logger.info(f"Stage 8 [Persistence]: Saving model artifact & metadata in {duration}s")

        model_version = ModelService.save_model_artifact(
            pipeline_id=pipeline_id,
            pipeline_name=pipe_name,
            task_type=task_type,
            model_name=model_name,
            model_obj=model,
            preprocessor_obj=preprocessor,
            feature_names=feature_names,
            metrics=eval_results["metrics"]
        )

        # Step 9: Log Experiment Run
        exp_id = ExperimentService.log_experiment_run(
            pipeline_id=pipeline_id,
            pipeline_name=pipe_name,
            dataset_id=dataset_id,
            model_version=model_version,
            algorithm=model_name,
            task_type=task_type,
            metrics=eval_results["metrics"],
            duration_seconds=duration
        )

        logger.info(f"Pipeline Run COMPLETED successfully! Run ID: {run_id}, Model Version: {model_version}")

        # Save run record
        run_record = {
            "run_id": run_id,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipe_name,
            "status": "COMPLETED",
            "model_version": model_version,
            "experiment_id": exp_id,
            "duration_seconds": duration,
            "timestamp": get_current_timestamp_iso(),
            "metrics": eval_results["metrics"],
            "logs": logger.get_logs()
        }
        PipelineService.save_pipeline_run(run_record)

        return run_record
