"""
MLForge - Pipeline Management Service Module
Handles saving, retrieving, and listing pipeline DAG configurations,
pipeline execution logs, and run history records.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from config import PIPELINE_DIR
from utils.files import load_json, save_json, delete_file, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class PipelineService:
    """
    Service for pipeline configuration persistence and execution history tracking.
    """
    
    @staticmethod
    def list_pipelines() -> List[Dict[str, Any]]:
        """
        Lists all saved pipeline configurations.
        """
        config_files = list_files_in_dir(PIPELINE_DIR, extension="json")
        pipelines = []
        
        for cfg_file in config_files:
            if not cfg_file.name.startswith("run_"):
                try:
                    pipeline_data = load_json(cfg_file)
                    pipelines.append(pipeline_data)
                except Exception:
                    continue
                    
        return sorted(pipelines, key=lambda x: x.get("created_at", ""), reverse=True)

    @staticmethod
    def get_pipeline(pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a saved pipeline configuration by ID.
        """
        cfg_path = PIPELINE_DIR / f"{pipeline_id}.json"
        if cfg_path.exists():
            return load_json(cfg_path)
            
        # Try matching by ID field inside JSON files
        config_files = list_files_in_dir(PIPELINE_DIR, extension="json")
        for cf in config_files:
            try:
                data = load_json(cf)
                if data.get("id") == pipeline_id:
                    return data
            except Exception:
                continue
                
        return None

    @staticmethod
    def save_pipeline(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves or updates a pipeline configuration JSON file.
        """
        pipeline_id = pipeline_config.get("id") or generate_unique_id("pipe")
        pipeline_config["id"] = pipeline_id
        
        if "created_at" not in pipeline_config:
            pipeline_config["created_at"] = get_current_timestamp_iso()
            
        pipeline_config["updated_at"] = get_current_timestamp_iso()
        
        cfg_path = PIPELINE_DIR / f"{pipeline_id}.json"
        save_json(pipeline_config, cfg_path)
        return pipeline_config

    @staticmethod
    def list_pipeline_runs() -> List[Dict[str, Any]]:
        """
        Lists all execution run records.
        """
        run_files = list_files_in_dir(PIPELINE_DIR, extension="json")
        runs = []
        
        for rf in run_files:
            if rf.name.startswith("run_"):
                try:
                    run_data = load_json(rf)
                    runs.append(run_data)
                except Exception:
                    continue
                    
        return sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)

    @staticmethod
    def save_pipeline_run(run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves a completed pipeline execution run artifact.
        """
        run_id = run_data.get("run_id") or generate_unique_id("run")
        run_data["run_id"] = run_id
        
        if "timestamp" not in run_data:
            run_data["timestamp"] = get_current_timestamp_iso()
            
        run_path = PIPELINE_DIR / f"run_{run_id}.json"
        save_json(run_data, run_path)
        return run_data

    @staticmethod
    def get_pipeline_run(run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific pipeline run record by ID.
        """
        run_path = PIPELINE_DIR / f"run_{run_id}.json"
        if run_path.exists():
            return load_json(run_path)
        return None
