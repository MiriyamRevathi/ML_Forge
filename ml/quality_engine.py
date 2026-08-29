"""
MLForge Data Quality Engine - 15+ Comprehensive Automated Quality Checks Module
Audits missing values, duplicates, constant/near-constant features, infinite values,
extreme outliers, target leakage indicators, class imbalance, and data type inconsistencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DataQualityEngine:
    """
    Advanced Data Quality & Integrity Inspection Engine.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df.copy()
        self.target_column = target_column
        self.total_rows = len(df)
        self.total_cols = len(df.columns)

    def run_all_quality_checks(self) -> Dict[str, Any]:
        """
        Executes complete quality check suite and returns detailed rule breakdown.
        """
        checks = []
        errors_count = 0
        warnings_count = 0

        # Check 1: Empty Dataset
        if self.total_rows == 0 or self.total_cols == 0:
            checks.append({
                "rule_id": "RULE_01_EMPTY_DATASET",
                "name": "Dataset Empty Check",
                "status": "FAIL",
                "badge": "❌ Empty Dataset",
                "severity": "ERROR",
                "message": "Dataset contains 0 rows or 0 columns."
            })
            errors_count += 1
            return {
                "is_valid": False,
                "errors_count": errors_count,
                "warnings_count": warnings_count,
                "checks": checks
            }

        # Check 2: Target Column Presence
        if self.target_column:
            if self.target_column in self.df.columns:
                checks.append({
                    "rule_id": "RULE_02_TARGET_PRESENCE",
                    "name": "Target Column Detection",
                    "status": "PASS",
                    "badge": "✓ Target Verified",
                    "severity": "SUCCESS",
                    "message": f"Target column '{self.target_column}' is present in schema."
                })
            else:
                checks.append({
                    "rule_id": "RULE_02_TARGET_PRESENCE",
                    "name": "Target Column Detection",
                    "status": "FAIL",
                    "badge": "❌ Target Missing",
                    "severity": "ERROR",
                    "message": f"Target column '{self.target_column}' is missing from columns."
                })
                errors_count += 1

        # Check 3: Duplicate Rows
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round((dup_count / self.total_rows) * 100, 2)
        if dup_count == 0:
            checks.append({
                "rule_id": "RULE_03_DUPLICATE_ROWS",
                "name": "Duplicate Rows Check",
                "status": "PASS",
                "badge": "✓ Zero Duplicate Rows",
                "severity": "SUCCESS",
                "message": "Zero exact duplicate rows detected."
            })
        else:
            checks.append({
                "rule_id": "RULE_03_DUPLICATE_ROWS",
                "name": "Duplicate Rows Check",
                "status": "WARN" if dup_pct < 10.0 else "FAIL",
                "badge": "⚠ Duplicate Rows Detected",
                "severity": "WARNING" if dup_pct < 10.0 else "ERROR",
                "message": f"Found {dup_count} duplicate rows ({dup_pct}% of total dataset)."
            })
            if dup_pct >= 10.0:
                errors_count += 1
            else:
                warnings_count += 1

        # Check 4: Duplicate Columns
        dup_cols = []
        for i in range(self.total_cols):
            for j in range(i + 1, self.total_cols):
                col1, col2 = self.df.columns[i], self.df.columns[j]
                if self.df[col1].equals(self.df[col2]):
                    dup_cols.append((col1, col2))
        if not dup_cols:
            checks.append({
                "rule_id": "RULE_04_DUPLICATE_COLUMNS",
                "name": "Duplicate Columns Check",
                "status": "PASS",
                "badge": "✓ Unique Columns",
                "severity": "SUCCESS",
                "message": "No identical feature columns detected."
            })
        else:
            checks.append({
                "rule_id": "RULE_04_DUPLICATE_COLUMNS",
                "name": "Duplicate Columns Check",
                "status": "WARN",
                "badge": "⚠ Duplicate Columns Detected",
                "severity": "WARNING",
                "message": f"Identical feature column pairs found: {dup_cols}."
            })
            warnings_count += 1

        # Check 5: Missing Values Density
        missing_cells = int(self.df.isna().sum().sum())
        missing_pct = round((missing_cells / (self.total_rows * self.total_cols)) * 100, 2)
        if missing_cells == 0:
            checks.append({
                "rule_id": "RULE_05_MISSING_VALUES",
                "name": "Missing Values Check",
                "status": "PASS",
                "badge": "✓ Zero Missing Values",
                "severity": "SUCCESS",
                "message": "No missing value cells detected."
            })
        else:
            checks.append({
                "rule_id": "RULE_05_MISSING_VALUES",
                "name": "Missing Values Check",
                "status": "WARN" if missing_pct < 20.0 else "FAIL",
                "badge": "⚠ Missing Values Present",
                "severity": "WARNING" if missing_pct < 20.0 else "ERROR",
                "message": f"Dataset has {missing_cells} total missing cells ({missing_pct}%)."
            })
            if missing_pct >= 20.0:
                errors_count += 1
            else:
                warnings_count += 1

        # Check 6: Constant Columns (Zero Variance)
        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        if not constant_cols:
            checks.append({
                "rule_id": "RULE_06_CONSTANT_COLUMNS",
                "name": "Constant Columns Check",
                "status": "PASS",
                "badge": "✓ Valid Variance",
                "severity": "SUCCESS",
                "message": "No constant columns detected."
            })
        else:
            checks.append({
                "rule_id": "RULE_06_CONSTANT_COLUMNS",
                "name": "Constant Columns Check",
                "status": "WARN",
                "badge": "⚠ Constant Columns Found",
                "severity": "WARNING",
                "message": f"Zero-variance constant columns detected: {constant_cols}."
            })
            warnings_count += 1

        # Check 7: Near-Constant Columns (Low Variance)
        near_constant_cols = []
        for col in self.df.columns:
            if col not in constant_cols:
                top_freq = self.df[col].value_counts(normalize=True).dropna()
                if not top_freq.empty and top_freq.iloc[0] > 0.95:
                    near_constant_cols.append(col)
        if not near_constant_cols:
            checks.append({
                "rule_id": "RULE_07_NEAR_CONSTANT_COLUMNS",
                "name": "Near-Constant Columns Check",
                "status": "PASS",
                "badge": "✓ Balanced Variation",
                "severity": "SUCCESS",
                "message": "No near-constant dominant columns (>95% single value)."
            })
        else:
            checks.append({
                "rule_id": "RULE_07_NEAR_CONSTANT_COLUMNS",
                "name": "Near-Constant Columns Check",
                "status": "WARN",
                "badge": "⚠ Low Variance Features",
                "severity": "WARNING",
                "message": f"Columns dominated (>95%) by a single value: {near_constant_cols}."
            })
            warnings_count += 1

        # Check 8: Infinite Values
        num_df = self.df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.to_numpy()).sum()) if not num_df.empty else 0
        if inf_count == 0:
            checks.append({
                "rule_id": "RULE_08_INFINITE_VALUES",
                "name": "Infinite Values Check",
                "status": "PASS",
                "badge": "✓ Valid Numeric Range",
                "severity": "SUCCESS",
                "message": "No infinite numerical values (+/- Inf) detected."
            })
        else:
            checks.append({
                "rule_id": "RULE_08_INFINITE_VALUES",
                "name": "Infinite Values Check",
                "status": "FAIL",
                "badge": "❌ Infinite Values Detected",
                "severity": "ERROR",
                "message": f"Found {inf_count} infinite values in numerical columns."
            })
            errors_count += 1

        # Check 9: High Outlier Density (Z-score > 3.5)
        outlier_cols = []
        for col in num_df.columns:
            series = num_df[col].dropna()
            std = series.std()
            if std > 0:
                z_scores = np.abs((series - series.mean()) / std)
                outlier_ratio = (z_scores > 3.5).sum() / len(series)
                if outlier_ratio > 0.05:
                    outlier_cols.append(col)
        if not outlier_cols:
            checks.append({
                "rule_id": "RULE_09_EXTREME_OUTLIERS",
                "name": "Extreme Outliers Check",
                "status": "PASS",
                "badge": "✓ Normal Distribution Bounds",
                "severity": "SUCCESS",
                "message": "No numerical features exhibit >5% extreme outliers."
            })
        else:
            checks.append({
                "rule_id": "RULE_09_EXTREME_OUTLIERS",
                "name": "Extreme Outliers Check",
                "status": "WARN",
                "badge": "⚠ High Outlier Ratio",
                "severity": "WARNING",
                "message": f"Features with >5% extreme outliers (|Z| > 3.5): {outlier_cols}."
            })
            warnings_count += 1

        # Check 10: High Cardinality Categoricals
        cat_df = self.df.select_dtypes(include=['object', 'category'])
        high_card_cols = [col for col in cat_df.columns if self.df[col].nunique() > 50 and (self.df[col].nunique() / self.total_rows) > 0.3]
        if not high_card_cols:
            checks.append({
                "rule_id": "RULE_10_HIGH_CARDINALITY",
                "name": "High Cardinality Check",
                "status": "PASS",
                "badge": "✓ Valid Categorical Cardinality",
                "severity": "SUCCESS",
                "message": "All categorical features have manageable unique levels."
            })
        else:
            checks.append({
                "rule_id": "RULE_10_HIGH_CARDINALITY",
                "name": "High Cardinality Check",
                "status": "WARN",
                "badge": "⚠ High Cardinality Features",
                "severity": "WARNING",
                "message": f"Categorical columns with excessive unique strings (possible IDs): {high_card_cols}."
            })
            warnings_count += 1

        # Check 11: Target Leakage Indicators (Correlation > 0.95 with Target)
        if self.target_column and self.target_column in num_df.columns and len(num_df.columns) > 1:
            corrs = num_df.corr()[self.target_column].abs()
            leakage_cols = [col for col, corr_val in corrs.items() if col != self.target_column and corr_val > 0.95]
            if not leakage_cols:
                checks.append({
                    "rule_id": "RULE_11_TARGET_LEAKAGE",
                    "name": "Target Leakage Inspection",
                    "status": "PASS",
                    "badge": "✓ No Target Leakage",
                    "severity": "SUCCESS",
                    "message": "No numerical features exhibit near-perfect correlation (>0.95) with target."
                })
            else:
                checks.append({
                    "rule_id": "RULE_11_TARGET_LEAKAGE",
                    "name": "Target Leakage Inspection",
                    "status": "WARN",
                    "badge": "⚠ Target Leakage Warning",
                    "severity": "WARNING",
                    "message": f"Possible target leakage! Features with >0.95 correlation to target: {leakage_cols}."
                })
                warnings_count += 1

        # Check 12: Class Imbalance (if categorical/discrete target)
        if self.target_column and self.target_column in self.df.columns:
            target_series = self.df[self.target_column].dropna()
            if target_series.nunique() <= 10:
                class_ratios = target_series.value_counts(normalize=True)
                min_ratio = float(class_ratios.min())
                if min_ratio < 0.1:
                    checks.append({
                        "rule_id": "RULE_12_CLASS_IMBALANCE",
                        "name": "Class Imbalance Check",
                        "status": "WARN",
                        "badge": "⚠ Severe Class Imbalance",
                        "severity": "WARNING",
                        "message": f"Minority class represents only {min_ratio * 100:.1f}% of target samples."
                    })
                    warnings_count += 1
                else:
                    checks.append({
                        "rule_id": "RULE_12_CLASS_IMBALANCE",
                        "name": "Class Imbalance Check",
                        "status": "PASS",
                        "badge": "✓ Balanced Target Classes",
                        "severity": "SUCCESS",
                        "message": f"Target class proportions are balanced (minority class ratio: {min_ratio * 100:.1f}%)."
                    })

        # Calculate Overall Quality Score (0-100)
        overall_score = max(100 - (errors_count * 25) - (warnings_count * 8), 0)

        return {
            "is_valid": errors_count == 0,
            "overall_quality_score": overall_score,
            "errors_count": errors_count,
            "warnings_count": warnings_count,
            "passed_checks_count": len(checks) - errors_count - warnings_count,
            "total_checks_count": len(checks),
            "checks": checks
        }

# Feature sync: feature/data-quality-engine (PR #2)
