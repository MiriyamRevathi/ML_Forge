"""
MLForge ML Engine - Data Validation & Quality Audit Module
Performs comprehensive data quality checks: empty data, target verification,
missing values, duplicate rows, infinite values, constant columns, high cardinality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class DataValidator:
    """
    Automated data quality & validation suite.
    """
    
    @staticmethod
    def validate_dataset(
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete quality validation checks on a Pandas DataFrame.
        """
        checks = []
        is_valid = True
        warnings_count = 0
        errors_count = 0
        
        # 1. Dataset Loaded check
        if df is None or df.empty:
            checks.append({
                "rule": "Dataset Loaded",
                "status": "FAIL",
                "badge": "❌ Empty Dataset",
                "message": "Dataset DataFrame is empty or None.",
                "severity": "ERROR"
            })
            errors_count += 1
            is_valid = False
            return {
                "is_valid": False,
                "score": 0,
                "errors_count": errors_count,
                "warnings_count": warnings_count,
                "checks": checks
            }
        else:
            checks.append({
                "rule": "Dataset Loaded",
                "status": "PASS",
                "badge": "✓ Dataset Loaded",
                "message": f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.",
                "severity": "SUCCESS"
            })

        # 2. Target Column Detection
        if target_column:
            if target_column in df.columns:
                checks.append({
                    "rule": "Target Column",
                    "status": "PASS",
                    "badge": "✓ Target Detected",
                    "message": f"Target column '{target_column}' confirmed in dataset schema.",
                    "severity": "SUCCESS"
                })
            else:
                checks.append({
                    "rule": "Target Column",
                    "status": "FAIL",
                    "badge": "❌ Missing Target",
                    "message": f"Specified target column '{target_column}' is missing from columns.",
                    "severity": "ERROR"
                })
                errors_count += 1
                is_valid = False
        else:
            checks.append({
                "rule": "Target Column",
                "status": "WARN",
                "badge": "⚠ Target Not Set",
                "message": "No target column specified. Defaulting to last column.",
                "severity": "WARNING"
            })
            warnings_count += 1

        # 3. Duplicate Rows Check
        dups_count = int(df.duplicated().sum())
        if dups_count == 0:
            checks.append({
                "rule": "Duplicate Rows",
                "status": "PASS",
                "badge": "✓ No Duplicate Rows",
                "message": "Zero duplicate rows found in dataset.",
                "severity": "SUCCESS"
            })
        else:
            checks.append({
                "rule": "Duplicate Rows",
                "status": "WARN",
                "badge": "⚠ Duplicate Rows Detected",
                "message": f"Found {dups_count} duplicate rows ({dups_count / len(df) * 100:.1f}%).",
                "severity": "WARNING"
            })
            warnings_count += 1

        # 4. Missing Values Check
        missing_count = int(df.isna().sum().sum())
        if missing_count == 0:
            checks.append({
                "rule": "Missing Values",
                "status": "PASS",
                "badge": "✓ No Missing Values",
                "message": "Zero missing cells found across all features.",
                "severity": "SUCCESS"
            })
        else:
            checks.append({
                "rule": "Missing Values",
                "status": "WARN",
                "badge": "⚠ Missing Values Detected",
                "message": f"Found {missing_count} total missing value cells.",
                "severity": "WARNING"
            })
            warnings_count += 1

        # 5. Infinite Values Check (Numerical)
        num_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.to_numpy()).sum()) if not num_df.empty else 0
        if inf_count == 0:
            checks.append({
                "rule": "Infinite Values",
                "status": "PASS",
                "badge": "✓ Valid Numerical Values",
                "message": "No positive or negative infinity values found in numerical columns.",
                "severity": "SUCCESS"
            })
        else:
            checks.append({
                "rule": "Infinite Values",
                "status": "FAIL",
                "badge": "❌ Infinite Values Detected",
                "message": f"Found {inf_count} infinite values in numerical columns.",
                "severity": "ERROR"
            })
            errors_count += 1
            is_valid = False

        # 6. Constant Columns (Zero Variance)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if not constant_cols:
            checks.append({
                "rule": "Constant Features",
                "status": "PASS",
                "badge": "✓ Feature Variance OK",
                "message": "No zero-variance constant columns detected.",
                "severity": "SUCCESS"
            })
        else:
            checks.append({
                "rule": "Constant Features",
                "status": "WARN",
                "badge": "⚠ Constant Columns",
                "message": f"Constant columns found with zero variance: {constant_cols}.",
                "severity": "WARNING"
            })
            warnings_count += 1

        # 7. High Cardinality Categoricals
        cat_df = df.select_dtypes(include=['object', 'category'])
        high_card_cols = [col for col in cat_df.columns if df[col].nunique() > 100 and df[col].nunique() > 0.5 * len(df)]
        if not high_card_cols:
            checks.append({
                "rule": "Categorical Cardinality",
                "status": "PASS",
                "badge": "✓ Valid Categorical Columns",
                "message": "All categorical columns have acceptable unique cardinality.",
                "severity": "SUCCESS"
            })
        else:
            checks.append({
                "rule": "Categorical Cardinality",
                "status": "WARN",
                "badge": "⚠ High Cardinality Features",
                "message": f"High cardinality columns detected (possible IDs): {high_card_cols}.",
                "severity": "WARNING"
            })
            warnings_count += 1

        # Overall Score Calculation (0-100)
        score = max(100 - (errors_count * 25) - (warnings_count * 10), 0)

        return {
            "is_valid": is_valid,
            "score": score,
            "errors_count": errors_count,
            "warnings_count": warnings_count,
            "checks": checks
        }
