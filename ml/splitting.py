"""
MLForge ML Engine - Train/Test Split Utility Module
Splits datasets into Training and Testing subsets with stratification,
random state seed control, and ratio customization (80/20, 70/30, 90/10).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Train/test dataset splitting utility.
    """
    
    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
        stratify: bool = True,
        task_type: str = "classification"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits DataFrame into X_train, X_test, y_train, y_test.
        """
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' does not exist in DataFrame.")
            
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()
        
        stratify_y = None
        if stratify and task_type == "classification":
            # Ensure minimum class count > 1 for stratification
            class_counts = y.value_counts()
            if (class_counts >= 2).all():
                stratify_y = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_y
        )
        
        return X_train, X_test, y_train, y_test
