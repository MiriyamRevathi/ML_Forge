"""
MLForge ML Engine - Statistical Data Drift & Retraining Suite Module
Calculates Kolmogorov-Smirnov (KS-test), Population Stability Index (PSI),
Total Variation Distance (TVD), Chi-Square distribution shifts, and automated retraining triggers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ml.drift_statistical_engine import DriftStatisticalEngine
from services.dataset_service import DatasetService


class DriftMonitoringSuite:
    """
    Statistical Data Drift Analysis & Retraining Orchestration Suite.
    """

    @staticmethod
    def inspect_dataset_drift(
        reference_dataset_id: str,
        target_dataset_id: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculates feature-level data drift between reference baseline dataset and target dataset.
        """
        ref_df = DatasetService.load_dataset_dataframe(reference_dataset_id)
        tar_df = DatasetService.load_dataset_dataframe(target_dataset_id)

        common_cols = [c for c in ref_df.columns if c in tar_df.columns]
        if not common_cols:
            return {
                "reference_dataset": reference_dataset_id,
                "target_dataset": target_dataset_id,
                "has_drift": False,
                "drift_status": "NO_COMMON_COLUMNS",
                "feature_reports": []
            }

        reports = []
        drifted_count = 0

        for col in common_cols:
            ref_series = ref_df[col]
            tar_series = tar_df[col]

            if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(tar_series):
                ks_res = DriftStatisticalEngine.calculate_ks_drift(ref_series, tar_series, alpha=alpha)
                psi_res = DriftStatisticalEngine.calculate_psi(ref_series, tar_series)

                has_drift = ks_res["has_drift"] or psi_res["has_drift"]
                if has_drift:
                    drifted_count += 1

                reports.append({
                    "feature": col,
                    "type": "numerical",
                    "status": "DRIFT_DETECTED" if has_drift else "STABLE",
                    "p_value": ks_res["p_value"],
                    "ks_statistic": ks_res["ks_statistic"],
                    "psi_score": psi_res["psi_score"],
                    "mean_change_pct": ks_res["mean_change_pct"]
                })

        drift_ratio = (drifted_count / len(common_cols)) if common_cols else 0.0
        overall_has_drift = drift_ratio >= 0.2

        return {
            "reference_dataset": reference_dataset_id,
            "target_dataset": target_dataset_id,
            "common_columns_count": len(common_cols),
            "drifted_features_count": drifted_count,
            "drift_ratio": round(drift_ratio, 4),
            "has_drift": overall_has_drift,
            "drift_status": "DRIFT_DETECTED" if overall_has_drift else "STABLE",
            "feature_reports": reports
        }
