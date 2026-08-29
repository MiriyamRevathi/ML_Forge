"""
MLForge - Prediction Service Module
Handles logging of single inference requests, dynamic input form payload parsing,
and storage of prediction history and batch CSV outputs.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from config import PREDICTION_DIR
from utils.files import load_json, save_json, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class PredictionService:
    """
    Service for single online prediction and batch prediction tracking.
    """
    
    @staticmethod
    def list_predictions() -> List[Dict[str, Any]]:
        """
        Lists past online prediction execution records.
        """
        pred_files = list_files_in_dir(PREDICTION_DIR, extension="json")
        predictions = []
        
        for pf in pred_files:
            try:
                pred_data = load_json(pf)
                predictions.append(pred_data)
            except Exception:
                continue
                
        return sorted(predictions, key=lambda x: x.get("timestamp", ""), reverse=True)

    @staticmethod
    def log_prediction(
        model_version: str,
        input_data: Dict[str, Any],
        prediction_result: Any,
        probabilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Logs a single online inference execution record.
        """
        pred_id = generate_unique_id("pred")
        
        record = {
            "id": pred_id,
            "model_version": model_version,
            "input_data": input_data,
            "prediction": prediction_result,
            "probabilities": probabilities,
            "timestamp": get_current_timestamp_iso()
        }
        
        pred_path = PREDICTION_DIR / f"{pred_id}.json"
        save_json(record, pred_path)
        return record

    @staticmethod
    def log_batch_prediction(
        model_version: str,
        original_filename: str,
        total_rows: int,
        processed_rows: int,
        output_csv_filename: str
    ) -> Dict[str, Any]:
        """
        Logs a batch prediction execution summary.
        """
        batch_id = generate_unique_id("batch")
        
        record = {
            "id": batch_id,
            "model_version": model_version,
            "original_filename": original_filename,
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "output_csv": output_csv_filename,
            "timestamp": get_current_timestamp_iso()
        }
        
        batch_path = PREDICTION_DIR / f"batch_{batch_id}.json"
        save_json(record, batch_path)
        return record
