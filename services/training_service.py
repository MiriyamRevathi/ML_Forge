"""
MLForge Services - Model Training & Cross-Validation Service Module
Handles algorithm hyperparameter schema querying, cross-validation runs,
Grid Search optimization dispatch, and model fitting.
"""

from typing import Dict, List, Any, Optional
from ml.models.classification import ClassificationCatalogue
from ml.models.regression import RegressionCatalogue
from ml.training import ModelTrainer
from ml.cross_validation import CrossValidationEngine
from ml.hyperparameter_tuning import HyperparameterTuner
from services.dataset_service import DatasetService


class TrainingService:
    """
    Business logic service for model training, CV, and hyperparameter tuning.
    """

    @staticmethod
    def get_supported_models_catalogue() -> Dict[str, Any]:
        """
        Returns full list of 10 classification algorithms and 10 regression algorithms with schemas.
        """
        classifiers = {}
        for key, name in ClassificationCatalogue.SUPPORTED_CLASSIFIERS.items():
            classifiers[key] = {
                "name": name,
                "schema": ClassificationCatalogue.get_hyperparameter_schema(key)
            }

        regressors = {}
        for key, name in RegressionCatalogue.SUPPORTED_REGRESSORS.items():
            regressors[key] = {
                "name": name,
                "schema": RegressionCatalogue.get_hyperparameter_schema(key)
            }

        return {
            "classification_models": classifiers,
            "regression_models": regressors
        }

    @staticmethod
    def run_model_cross_validation(
        dataset_id: str,
        model_name: str,
        task_type: str,
        n_splits: int = 5,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes N-Fold Cross-Validation on requested dataset using specified model architecture.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            raise FileNotFoundError(f"Dataset ID '{dataset_id}' not found.")

        df = DatasetService.load_dataset_dataframe(dataset_id)
        target_col = meta.get("target_column")
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        # Prepare X, y
        X = df.drop(columns=[target_col]).select_dtypes(include=['number']).fillna(0).values
        y = df[target_col].values

        if task_type == "classification":
            model = ClassificationCatalogue.create_classifier(model_name, hyperparameters)
        else:
            model = RegressionCatalogue.create_regressor(model_name, hyperparameters)

        cv_results = CrossValidationEngine.run_cross_validation(
            model=model,
            task_type=task_type,
            X=X,
            y=y,
            n_splits=n_splits
        )

        return {
            "dataset_name": meta.get("name"),
            "model_name": model_name,
            "task_type": task_type,
            "cv_results": cv_results
        }
