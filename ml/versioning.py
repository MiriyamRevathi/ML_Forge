"""
MLForge ML Engine - Model Versioning & Artifact Packaging Module
Packages trained models, preprocessors, and metadata into versioned disk artifacts.
"""

from typing import Dict, List, Any, Optional
from services.model_service import ModelService


class ModelVersionManager:
    """
    Model versioning artifact packager.
    """
    
    @staticmethod
    def create_version(
        model_object: Any,
        model_name: str,
        task_type: str,
        dataset_name: str,
        target_column: str,
        feature_names: List[str],
        metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        preprocessor_object: Optional[Any] = None,
        pipeline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registers model artifact version using ModelService.
        """
        metadata = ModelService.register_model(
            model_object=model_object,
            model_name=model_name,
            task_type=task_type,
            dataset_name=dataset_name,
            target_column=target_column,
            feature_names=feature_names,
            metrics=metrics,
            hyperparameters=hyperparameters,
            pipeline_id=pipeline_id,
            preprocessor_object=preprocessor_object
        )
        return metadata
