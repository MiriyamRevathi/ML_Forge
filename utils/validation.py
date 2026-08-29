"""
MLForge - Input & Request Validation Utility Module
Provides defensive validation methods for REST requests, dataset parameters,
pipeline specifications, and hyperparameter configurations.
"""

from typing import Dict, Any, List, Tuple, Optional
from config import SUPPORTED_TASKS, CLASSIFICATION_MODELS, REGRESSION_MODELS


class ValidationError(Exception):
    """Exception thrown when request or dataset validation checks fail."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def validate_dataset_upload_request(filename: str, file_size: int) -> bool:
    """
    Validates file extension and size before processing dataset uploads.
    """
    if not filename:
        raise ValidationError("No file provided in request.")
        
    if not (filename.endswith('.csv') or filename.endswith('.json')):
        raise ValidationError("Only CSV (.csv) and JSON (.json) dataset files are supported.")
        
    if file_size <= 0:
        raise ValidationError("Uploaded file is empty (0 bytes).")
        
    return True


def validate_pipeline_configuration(config: Dict[str, Any]) -> bool:
    """
    Validates the structure and parameter integrity of a pipeline configuration JSON payload.
    """
    required_keys = ["name", "dataset_id", "target_column", "task"]
    for key in required_keys:
        if key not in config or not config[key]:
            raise ValidationError(f"Missing required pipeline configuration field: '{key}'.")
            
    task = config["task"]
    if task not in SUPPORTED_TASKS:
        raise ValidationError(f"Task '{task}' is not supported. Choose from {SUPPORTED_TASKS}.")
        
    # Validate model selection if provided
    model_name = config.get("model", {}).get("name")
    if model_name:
        valid_models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        if model_name not in valid_models:
            raise ValidationError(
                f"Model '{model_name}' is invalid for task '{task}'. Valid choices: {list(valid_models.keys())}."
            )
            
    # Validate test size split ratio
    test_size = config.get("test_size", 0.2)
    try:
        test_size_float = float(test_size)
        if not (0.05 <= test_size_float <= 0.5):
            raise ValidationError(f"Test size split ratio must be between 0.05 and 0.5 (received {test_size}).")
    except (ValueError, TypeError):
        raise ValidationError("Test size must be a valid float value.")
        
    return True


def sanitize_input_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes string inputs in dictionary payloads to prevent cross-site scripting or injection.
    """
    sanitized = {}
    for key, value in input_dict.items():
        if isinstance(value, str):
            sanitized[key] = value.strip()
        elif isinstance(value, dict):
            sanitized[key] = sanitize_input_dict(value)
        else:
            sanitized[key] = value
    return sanitized
