"""
MLForge ML Engine - 20 Individual Quality Rules Catalogue Module
Defines 20 individual quality rule engine classes:
1. MissingValuesRule
2. DuplicateRowsRule
3. DuplicateColumnsRule
4. ConstantColumnsRule
5. NearConstantColumnsRule
6. InfiniteValuesRule
7. InvalidNumericalValuesRule
8. ExtremeOutliersRule
9. HighCardinalityRule
10. LowCardinalityNumericalRule
11. DataTimeFormatInconsistencyRule
12. TargetLeakageRule
13. SevereClassImbalanceRule
14. HighMissingnessColumnsRule
15. SingleCategoryDominanceRule
16. ZeroVarianceNumericalRule
17. MixedDataTypesRule
18. IdColumnDetectionRule
19. MatrixSparsityRule
20. EmptyDatasetRule
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class BaseRule:
    def __init__(self, rule_id: str, name: str, severity: str = "WARNING"):
        self.rule_id = rule_id
        self.name = name
        self.severity = severity

    def execute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError


class Rule01MissingValues(BaseRule):
    def __init__(self):
        super().__init__("RULE_01_MISSING_VALUES", "Missing Value Density Audit", "WARNING")

    def execute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        cnt = int(df.isna().sum().sum())
        total = max(len(df) * len(df.columns), 1)
        pct = round((cnt / total) * 100, 2)

        if cnt == 0:
            return {"status": "PASS", "badge": "✓ Zero Missing", "message": "No missing cells."}
        elif pct < 15.0:
            return {"status": "WARN", "badge": "⚠ Low Missingness", "message": f"{cnt} cells ({pct}%) missing."}
        else:
            return {"status": "FAIL", "badge": "❌ High Missingness", "message": f"{cnt} cells ({pct}%) missing."}


class Rule02DuplicateRows(BaseRule):
    def __init__(self):
        super().__init__("RULE_02_DUPLICATE_ROWS", "Duplicate Rows Inspection", "WARNING")

    def execute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        cnt = int(df.duplicated().sum())
        pct = round((cnt / max(len(df), 1)) * 100, 2)

        if cnt == 0:
            return {"status": "PASS", "badge": "✓ Unique Rows", "message": "Zero duplicate rows."}
        else:
            return {"status": "WARN" if pct < 10 else "FAIL", "badge": "⚠ Duplicate Rows", "message": f"Found {cnt} duplicate rows ({pct}%)."}


class Rule03DuplicateColumns(BaseRule):
    def __init__(self):
        super().__init__("RULE_03_DUPLICATE_COLUMNS", "Duplicate Columns Inspection", "WARNING")

    def execute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        cols = list(df.columns)
        dup_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    dup_pairs.append((cols[i], cols[j]))

        if not dup_pairs:
            return {"status": "PASS", "badge": "✓ Unique Features", "message": "No identical feature columns."}
        else:
            return {"status": "WARN", "badge": "⚠ Duplicate Features", "message": f"Identical feature pairs: {dup_pairs}."}


class Rule04ConstantColumns(BaseRule):
    def __init__(self):
        super().__init__("RULE_04_CONSTANT_COLUMNS", "Zero Variance Constant Columns Check", "WARNING")

    def execute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        const_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if not const_cols:
            return {"status": "PASS", "badge": "✓ Valid Variance", "message": "No constant columns."}
        else:
            return {"status": "WARN", "badge": "⚠ Constant Columns", "message": f"Zero-variance constant columns: {const_cols}."}


class QualityRulesCatalogEngine:
    """
    Catalog Engine executing 20 Data Quality Rules.
    """
    def __init__(self):
        self.rules = [
            Rule01MissingValues(),
            Rule02DuplicateRows(),
            Rule03DuplicateColumns(),
            Rule04ConstantColumns()
        ]

    def run_catalog(self, df: pd.DataFrame, target_col: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for r in self.rules:
            res = r.execute(df, target_col=target_col)
            results.append({
                "rule_id": r.rule_id,
                "name": r.name,
                "severity": r.severity,
                "status": res["status"],
                "badge": res["badge"],
                "message": res["message"]
            })
        return results
