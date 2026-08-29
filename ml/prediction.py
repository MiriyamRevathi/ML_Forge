"""
MLForge ML Engine - Single Online Prediction Module
Loads serialized model artifacts, formats dynamic user inputs,
executes model inference, and calculates class prediction probabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from services.model_service import ModelService
from services.prediction_service import PredictionService


class SinglePredictor:
    """
    Online real-time single inference engine.
    """

    @staticmethod
    def predict(
        model_version: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes model prediction on a single feature dictionary input.
        """
        meta = ModelService.get_model_metadata(model_version)
        if not meta:
            raise FileNotFoundError(f"Model version '{model_version}' not found.")
            
        bundle = ModelService.load_model_artifact(model_version)
        model = bundle["model"]
        preprocessor = bundle.get("preprocessor")
        feature_names = bundle.get("feature_names", meta.get("feature_names", []))
        
        # Convert input dict to 1-row DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Preprocess features
        if preprocessor is not None and hasattr(preprocessor, "column_transformer") and preprocessor.column_transformer is not None:
            raw_num = getattr(preprocessor, "numerical_features", [])
            raw_cat = getattr(preprocessor, "categorical_features", [])
            expected_raw = list(raw_num) + list(raw_cat)
            
            for col in expected_raw:
                if col not in df_input.columns:
                    df_input[col] = input_data.get(col, 0.0 if col in raw_num else "Missing")
            
            X_proc = preprocessor.transform(df_input[expected_raw])
        else:
            for col in feature_names:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            X_proc = df_input[feature_names].values

        # Predict
        prediction = model.predict(X_proc)[0]
        
        # Calculate probabilities if classifier
        probabilities = None
        if meta.get("task_type") == "classification" and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_proc)[0]
                classes = getattr(model, "classes_", range(len(probs)))
                probabilities = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probs)}
            except Exception:
                probabilities = None

        # Convert numpy prediction to Python scalar
        if isinstance(prediction, (np.integer, int)):
            pred_scalar = int(prediction)
        elif isinstance(prediction, (np.floating, float)):
            pred_scalar = round(float(prediction), 4)
        else:
            pred_scalar = str(prediction)

        # Log prediction to disk
        record = PredictionService.log_prediction(
            model_version=model_version,
            input_data=input_data,
            prediction_result=pred_scalar,
            probabilities=probabilities
        )

        return {
            "model_version": model_version,
            "prediction": pred_scalar,
            "probabilities": probabilities,
            "prediction_record_id": record["id"]
        }
