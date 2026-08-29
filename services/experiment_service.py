"""
MLForge - Experiment Tracking Service Module
Tracks machine learning experiment iterations, metrics, hyperparameters,
pipeline association, and model artifact links.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from config import EXPERIMENT_DIR
from utils.files import load_json, save_json, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class ExperimentService:
    """
    Service for experiment run registration, indexing, and comparison.
    """
    
    @staticmethod
    def list_experiments() -> List[Dict[str, Any]]:
        """
        Lists all recorded experiments sorted by date.
        """
        exp_files = list_files_in_dir(EXPERIMENT_DIR, extension="json")
        experiments = []
        
        for ef in exp_files:
            try:
                exp_data = load_json(ef)
                experiments.append(exp_data)
            except Exception:
                continue
                
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)

    @staticmethod
    def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves experiment details by ID.
        """
        exp_path = EXPERIMENT_DIR / f"{experiment_id}.json"
        if exp_path.exists():
            return load_json(exp_path)
            
        exp_files = list_files_in_dir(EXPERIMENT_DIR, extension="json")
        for ef in exp_files:
            try:
                data = load_json(ef)
                if data.get("id") == experiment_id:
                    return data
            except Exception:
                continue
                
        return None

    @staticmethod
    def log_experiment(
        name: str,
        dataset_name: str,
        target: str,
        task: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        training_duration_seconds: float,
        model_version: Optional[str] = None,
        pipeline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates and logs a complete experiment run record.
        """
        exp_id = generate_unique_id("exp")
        
        exp_data = {
            "id": exp_id,
            "name": name,
            "dataset": dataset_name,
            "target": target,
            "task": task,
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "training_duration": training_duration_seconds,
            "model_version": model_version,
            "pipeline_id": pipeline_id,
            "status": "COMPLETED",
            "created_at": get_current_timestamp_iso()
        }
        
        exp_path = EXPERIMENT_DIR / f"{exp_id}.json"
        save_json(exp_data, exp_path)
        return exp_data
