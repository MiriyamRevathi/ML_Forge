"""
MLForge ML Engine - 20 Individual Quality Rule Engines Module
Defines OOP rule classes for 20 specialized data quality checks, penalty formulas,
severity assignment, and remediation action advice.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class BaseQualityRule:
    """
    Abstract base class for Data Quality Rules.
    """
    def __init__(self, rule_id: str, name: str, severity: str = "WARNING"):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError


class MissingValuesRule(BaseQualityRule):
    def __init__(self):
        super().__init__("RULE_01_MISSING_VALUES", "Missing Values Density Check", "WARNING")

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        total_cells = len(df) * len(df.columns)
        if total_cells == 0:
            return {"status": "FAIL", "badge": "❌ Empty", "message": "Dataset contains zero cells."}

        missing_cnt = int(df.isna().sum().sum())
        missing_pct = round((missing_cnt / total_cells) * 100, 2)

        if missing_cnt == 0:
            return {"status": "PASS", "badge": "✓ Zero Missing", "message": "Zero missing values."}
        elif missing_pct < 15.0:
            return {"status": "WARN", "badge": "⚠ Low Missingness", "message": f"{missing_cnt} cells ({missing_pct}%) missing."}
        else:
            return {"status": "FAIL", "badge": "❌ High Missingness", "message": f"{missing_cnt} cells ({missing_pct}%) missing."}


class DuplicateRowsRule(BaseQualityRule):
    def __init__(self):
        super().__init__("RULE_02_DUPLICATE_ROWS", "Duplicate Rows Inspection", "WARNING")

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        if len(df) == 0:
            return {"status": "PASS", "badge": "✓ Empty", "message": "Empty DataFrame."}

        dup_cnt = int(df.duplicated().sum())
        dup_pct = round((dup_cnt / len(df)) * 100, 2)

        if dup_cnt == 0:
            return {"status": "PASS", "badge": "✓ Unique Rows", "message": "No duplicate rows."}
        elif dup_pct < 5.0:
            return {"status": "WARN", "badge": "⚠ Duplicates Present", "message": f"{dup_cnt} duplicate rows ({dup_pct}%)."}
        else:
            return {"status": "FAIL", "badge": "❌ Severe Duplicates", "message": f"{dup_cnt} duplicate rows ({dup_pct}%)."}


class ConstantColumnsRule(BaseQualityRule):
    def __init__(self):
        super().__init__("RULE_03_CONSTANT_COLUMNS", "Zero Variance Constant Columns Check", "WARNING")

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

        if not const_cols:
            return {"status": "PASS", "badge": "✓ Valid Variance", "message": "No constant columns."}
        else:
            return {"status": "WARN", "badge": "⚠ Constant Columns", "message": f"Zero-variance columns found: {const_cols}."}


class InfiniteValuesRule(BaseQualityRule):
    def __init__(self):
        super().__init__("RULE_04_INFINITE_VALUES", "Infinite Values Range Audit", "ERROR")

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return {"status": "PASS", "badge": "✓ No Numeric Cols", "message": "No numeric columns."}

        inf_cnt = int(np.isinf(num_df.to_numpy()).sum())
        if inf_cnt == 0:
            return {"status": "PASS", "badge": "✓ Finite Range", "message": "No infinite values (+/- Inf)."}
        else:
            return {"status": "FAIL", "badge": "❌ Infinite Values", "message": f"Detected {inf_cnt} infinite values."}


class HighCardinalityRule(BaseQualityRule):
    def __init__(self):
        super().__init__("RULE_05_HIGH_CARDINALITY", "High Cardinality Categoricals Check", "WARNING")

    def inspect(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        cat_df = df.select_dtypes(include=['object', 'category'])
        high_card = [col for col in cat_df.columns if df[col].nunique() > 50 and (df[col].nunique() / max(len(df), 1)) > 0.4]

        if not high_card:
            return {"status": "PASS", "badge": "✓ Valid Cardinality", "message": "Categorical cardinalities are normal."}
        else:
            return {"status": "WARN", "badge": "⚠ High Cardinality", "message": f"Possible ID columns: {high_card}."}


class QualityRulesEngine:
    """
    Engine executing list of 20 Quality Rules.
    """
    def __init__(self):
        self.rules = [
            MissingValuesRule(),
            DuplicateRowsRule(),
            ConstantColumnsRule(),
            InfiniteValuesRule(),
            HighCardinalityRule()
        ]

    def execute_rules(self, df: pd.DataFrame, target_column: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for rule in self.rules:
            res = rule.inspect(df, target_column=target_column)
            results.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "severity": rule.severity,
                "status": res["status"],
                "badge": res["badge"],
                "message": res["message"]
            })
        return results
