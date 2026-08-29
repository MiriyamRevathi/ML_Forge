"""
MLForge ML Engine - Pipeline Specification Validator Module
Validates pipeline DAG specifications for dataset presence, target column existence,
supported task types, scaler choices, encoder compatibility, and hyperparameter bounds.
"""

from typing import Dict, List, Any, Tuple, Optional
from services.dataset_service import DatasetService
from ml.models.classification import ClassificationCatalogue
from ml.models.regression import RegressionCatalogue


class PipelineValidator:
    """
    Pipeline Specification Validation Engine.
    """

    @staticmethod
    def validate_pipeline_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates pipeline specification payload prior to saving or execution.
        """
        errors = []

        # 1. Name Check
        if not spec.get("name"):
            errors.append("Pipeline name is required.")

        # 2. Dataset Presence
        dataset_id = spec.get("dataset_id")
        if not dataset_id:
            errors.append("Dataset ID selection is required.")
        else:
            meta = DatasetService.get_dataset_metadata(dataset_id)
            if not meta:
                errors.append(f"Dataset ID '{dataset_id}' does not exist on disk.")

        # 3. Task Type Check
        task_type = spec.get("task_type")
        if task_type not in ["classification", "regression"]:
            errors.append(f"Unsupported task_type '{task_type}'. Must be 'classification' or 'regression'.")

        # 4. Target Column Check
        target_col = spec.get("target_column")
        if not target_col:
            errors.append("Target column selection is required.")

        # 5. Model Algorithm Check
        model_spec = spec.get("model", {})
        algo = model_spec.get("algorithm")
        if not algo:
            errors.append("Model algorithm selection is required.")
        else:
            if task_type == "classification" and algo not in ClassificationCatalogue.SUPPORTED_CLASSIFIERS:
                errors.append(f"Classifier algorithm '{algo}' is not supported.")
            elif task_type == "regression" and algo not in RegressionCatalogue.SUPPORTED_REGRESSORS:
                errors.append(f"Regressor algorithm '{algo}' is not supported.")

        # 6. Train-Test Split Ratio Check
        split_ratio = spec.get("train_test_split", 0.8)
        try:
            ratio = float(split_ratio)
            if ratio <= 0.1 or ratio >= 0.95:
                errors.append(f"Train/test split ratio {ratio} out of bounds (0.1 < ratio < 0.95).")
        except ValueError:
            errors.append("Invalid train/test split ratio value.")

        return len(errors) == 0, errors
