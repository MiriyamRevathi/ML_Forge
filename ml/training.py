"""
MLForge ML Engine - Model Training Module
Instantiates, configures, validates hyperparameters, and trains scikit-learn models
for Classification and Regression tasks.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class ModelTrainer:
    """
    Scikit-learn model instantiation, hyperparameter validation, and model fitting engine.
    """

    @staticmethod
    def get_classifier(model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Instantiates a classification model with specified hyperparameters.
        """
        params = params or {}
        name = model_name.lower().replace(" ", "_")
        
        if name in ["logistic_regression", "logisticregression"]:
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 500)),
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["decision_tree", "decisiontreeclassifier"]:
            return DecisionTreeClassifier(
                max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["random_forest", "randomforestclassifier"]:
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=int(params.get("random_state", 42)),
                n_jobs=-1
            )
        elif name in ["knn", "kneighborsclassifier"]:
            return KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 5)),
                weights=str(params.get("weights", "uniform"))
            )
        elif name in ["svm", "svc"]:
            return SVC(
                C=float(params.get("C", 1.0)),
                kernel=str(params.get("kernel", "rbf")),
                probability=True,
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["naive_bayes", "gaussiannb"]:
            return GaussianNB()
        elif name in ["gradient_boosting", "gradientboostingclassifier"]:
            return GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=int(params.get("random_state", 42))
            )
        else:
            # Default fallback to Random Forest
            return RandomForestClassifier(n_estimators=100, random_state=42)

    @staticmethod
    def get_regressor(model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Instantiates a regression model with specified hyperparameters.
        """
        params = params or {}
        name = model_name.lower().replace(" ", "_")
        
        if name in ["linear_regression", "linearregression"]:
            return LinearRegression()
        elif name in ["ridge"]:
            return Ridge(
                alpha=float(params.get("alpha", 1.0)),
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["lasso"]:
            return Lasso(
                alpha=float(params.get("alpha", 1.0)),
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["decision_tree_regressor", "decisiontreeregressor"]:
            return DecisionTreeRegressor(
                max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=int(params.get("random_state", 42))
            )
        elif name in ["random_forest_regressor", "randomforestregressor"]:
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=int(params.get("random_state", 42)),
                n_jobs=-1
            )
        elif name in ["gradient_boosting_regressor", "gradientboostingregressor"]:
            return GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=int(params.get("random_state", 42))
            )
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    @staticmethod
    def train_model(
        task_type: str,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Instantiates and fits model on training data.
        """
        if task_type == "classification":
            model = ModelTrainer.get_classifier(model_name, hyperparameters)
        else:
            model = ModelTrainer.get_regressor(model_name, hyperparameters)
            
        model.fit(X_train, y_train)
        return model

# Feature sync: feature/model-training-catalogue (PR #7)
