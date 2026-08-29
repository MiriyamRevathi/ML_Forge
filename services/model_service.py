"""
MLForge - Model Registry & Artifact Service Module
Manages model binaries (.joblib), metadata records (.json), version lifecycle states
(TRAINED, VALIDATED, STAGING, PRODUCTION, ARCHIVED), state transitions, and retrieval.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from config import (
    MODEL_DIR,
    STATUS_TRAINED,
    STATUS_PRODUCTION,
    REGISTRY_STATUSES
)
from utils.files import load_json, save_json, load_joblib, save_joblib, delete_file, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class ModelService:
    """
    Service for model binary persistence, model versioning, and registry lifecycle management.
    """
    
    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """
        Lists all registered models with version and lifecycle status.
        """
        meta_files = list_files_in_dir(MODEL_DIR, extension="json")
        models = []
        
        for mf in meta_files:
            try:
                meta = load_json(mf)
                models.append(meta)
            except Exception:
                continue
                
        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)

    @staticmethod
    def get_model_metadata(model_version: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata JSON for a model version.
        """
        meta_path = MODEL_DIR / f"{model_version}_meta.json"
        if meta_path.exists():
            return load_json(meta_path)
            
        # Try matching by ID or version field
        meta_files = list_files_in_dir(MODEL_DIR, extension="json")
        for mf in meta_files:
            try:
                meta = load_json(mf)
                if meta.get("version") == model_version or meta.get("id") == model_version:
                    return meta
            except Exception:
                continue
                
        return None

    @staticmethod
    def get_production_model(task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Returns the active PRODUCTION model metadata for a given task type (or latest production model).
        """
        all_models = ModelService.list_models()
        prod_models = [m for m in all_models if m.get("status") == STATUS_PRODUCTION]
        
        if task_type:
            prod_models = [m for m in prod_models if m.get("task_type") == task_type or m.get("task") == task_type]
            
        if prod_models:
            return prod_models[0]
            
        return None

    @staticmethod
    def load_model_artifact(model_version: str) -> Any:
        """
        Loads the joblib serialized scikit-learn model object for a model version.
        """
        meta = ModelService.get_model_metadata(model_version)
        if not meta:
            raise FileNotFoundError(f"Model metadata for version '{model_version}' not found.")
            
        binary_filename = meta.get("artifact_filename", f"{model_version}.joblib")
        binary_path = MODEL_DIR / binary_filename
        
        if not binary_path.exists():
            raise FileNotFoundError(f"Model binary artifact '{binary_filename}' not found on disk.")
            
        return load_joblib(binary_path)

    @staticmethod
    def register_model(
        model_object: Any,
        model_name: str,
        task_type: str,
        dataset_name: str,
        target_column: str,
        feature_names: List[str],
        metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        pipeline_id: Optional[str] = None,
        preprocessor_object: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Registers a new trained model artifact and associated metadata.
        """
        # Determine model version count
        existing = ModelService.list_models()
        version_num = len(existing) + 1
        version_id = f"model_v{version_num}"
        
        binary_filename = f"{version_id}.joblib"
        binary_path = MODEL_DIR / binary_filename
        
        # Save model bundle (model + optional preprocessor)
        artifact_bundle = {
            "model": model_object,
            "preprocessor": preprocessor_object,
            "feature_names": feature_names
        }
        save_joblib(artifact_bundle, binary_path)
        
        metadata = {
            "id": version_id,
            "version": version_id,
            "name": model_name,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "pipeline_id": pipeline_id,
            "status": STATUS_TRAINED,
            "artifact_filename": binary_filename,
            "created_at": get_current_timestamp_iso()
        }
        
        meta_path = MODEL_DIR / f"{version_id}_meta.json"
        save_json(metadata, meta_path)
        return metadata

    @staticmethod
    def update_model_status(model_version: str, new_status: str) -> Optional[Dict[str, Any]]:
        """
        Updates model lifecycle status (e.g. promoting to PRODUCTION or archiving).
        """
        if new_status not in REGISTRY_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Choose from {REGISTRY_STATUSES}")
            
        meta = ModelService.get_model_metadata(model_version)
        if not meta:
            return None
            
        # If promoting to PRODUCTION, demote previous PRODUCTION model of same task_type
        if new_status == STATUS_PRODUCTION:
            all_models = ModelService.list_models()
            task_type = meta.get("task_type")
            for m in all_models:
                if m.get("status") == STATUS_PRODUCTION and m.get("task_type") == task_type and m.get("version") != model_version:
                    m["status"] = "STAGING"
                    save_json(m, MODEL_DIR / f"{m['version']}_meta.json")
                    
        meta["status"] = new_status
        meta["updated_at"] = get_current_timestamp_iso()
        
        meta_path = MODEL_DIR / f"{meta['version']}_meta.json"
        save_json(meta, meta_path)
        return meta
