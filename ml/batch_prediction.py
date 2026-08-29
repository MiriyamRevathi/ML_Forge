"""
MLForge ML Engine - Bulk CSV Batch Prediction Module
Processes batch CSV uploads, runs inference over thousands of rows,
appends prediction columns, and generates downloadable output CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from config import PREDICTION_DIR
from services.model_service import ModelService
from services.prediction_service import PredictionService
from utils.helpers import generate_unique_id


class BatchPredictor:
    """
    Bulk CSV inference processor.
    """

    @staticmethod
    def process_batch_csv(
        model_version: str,
        input_csv_path: Path,
        original_filename: str
    ) -> Dict[str, Any]:
        """
        Processes batch CSV file and produces predictions output CSV.
        """
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input batch CSV file '{input_csv_path}' not found.")
            
        meta = ModelService.get_model_metadata(model_version)
        if not meta:
            raise FileNotFoundError(f"Model version '{model_version}' not found.")
            
        bundle = ModelService.load_model_artifact(model_version)
        model = bundle["model"]
        preprocessor = bundle.get("preprocessor")
        feature_names = bundle.get("feature_names", meta.get("feature_names", []))
        
        df_batch = pd.read_csv(input_csv_path)
        total_rows = len(df_batch)
        
        if total_rows == 0:
            raise ValueError("Uploaded batch CSV file contains zero data rows.")

        # Save copy of features for transformation
        df_features = df_batch.copy()
        
        # Drop target column if present in batch file
        target_col = meta.get("target_column")
        if target_col and target_col in df_features.columns:
            df_features = df_features.drop(columns=[target_col])
            
        if preprocessor is not None:
            try:
                X_proc = preprocessor.transform(df_features)
            except Exception as e:
                # Fallback: select expected numerical & categorical columns
                num_cols = preprocessor.numerical_features
                cat_cols = preprocessor.categorical_features
                for col in num_cols:
                    if col not in df_features.columns:
                        df_features[col] = 0.0
                for col in cat_cols:
                    if col not in df_features.columns:
                        df_features[col] = "Missing"
                X_proc = preprocessor.transform(df_features)
        else:
            for col in feature_names:
                if col not in df_features.columns:
                    df_features[col] = 0.0
            X_proc = df_features[feature_names].values

        # Run inference
        predictions = model.predict(X_proc)
        
        # Append predictions to output DataFrame
        df_output = df_batch.copy()
        df_output["prediction"] = predictions
        
        # Append probabilities if classification model
        if meta.get("task_type") == "classification" and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_proc)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    df_output["prediction_probability"] = np.round(probs[:, 1], 4)
            except Exception:
                pass

        # Save output CSV to predictions storage
        batch_id = generate_unique_id("batch_out")
        output_filename = f"predictions_{batch_id}.csv"
        output_filepath = PREDICTION_DIR / output_filename
        
        df_output.to_csv(output_filepath, index=False)
        
        # Log batch run summary
        record = PredictionService.log_batch_prediction(
            model_version=model_version,
            original_filename=original_filename,
            total_rows=total_rows,
            processed_rows=len(df_output),
            output_csv_filename=output_filename
        )
        
        return {
            "batch_id": record["id"],
            "model_version": model_version,
            "total_rows": total_rows,
            "processed_rows": len(df_output),
            "output_filename": output_filename,
            "download_url": f"/predictions/download/{output_filename}"
        }
