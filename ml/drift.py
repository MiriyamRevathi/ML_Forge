"""
MLForge ML Engine - Data Drift Detection Module
Performs statistical distribution drift analysis comparing reference dataset (training baseline)
against target dataset using Kolmogorov-Smirnov (KS-test), mean/std shifts, and category frequencies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from services.dataset_service import DatasetService
from services.monitoring_service import MonitoringService
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class DataDriftDetector:
    """
    Statistical Data Drift Engine.
    """

    @staticmethod
    def detect_drift(
        reference_dataset_id: str,
        target_dataset_id: str,
        ks_alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculates feature-level data drift metrics comparing reference and target datasets.
        """
        ref_df = DatasetService.load_dataset_dataframe(reference_dataset_id)
        tar_df = DatasetService.load_dataset_dataframe(target_dataset_id)
        
        feature_reports = []
        drifted_count = 0
        warning_count = 0
        
        common_cols = [c for c in ref_df.columns if c in tar_df.columns]
        
        for col in common_cols:
            ref_series = ref_df[col].dropna()
            tar_series = tar_df[col].dropna()
            
            if ref_series.empty or tar_series.empty:
                continue
                
            if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(tar_series):
                # 1. Numerical Feature Drift via Kolmogorov-Smirnov 2-sample test
                ks_stat, p_value = stats.ks_2samp(ref_series, tar_series)
                
                ref_mean = float(ref_series.mean())
                tar_mean = float(tar_series.mean())
                ref_std = float(ref_series.std()) if len(ref_series) > 1 else 1.0
                tar_std = float(tar_series.std()) if len(tar_series) > 1 else 1.0
                
                mean_diff_pct = abs((tar_mean - ref_mean) / (ref_mean if ref_mean != 0 else 1.0)) * 100
                
                # Determine drift status
                if p_value < ks_alpha and mean_diff_pct > 15.0:
                    status = "DRIFT DETECTED"
                    drifted_count += 1
                elif p_value < ks_alpha or mean_diff_pct > 10.0:
                    status = "WARNING"
                    warning_count += 1
                else:
                    status = "NORMAL"

                feature_reports.append({
                    "feature": col,
                    "type": "numerical",
                    "status": status,
                    "p_value": round(float(p_value), 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "reference_mean": round(ref_mean, 4),
                    "target_mean": round(tar_mean, 4),
                    "mean_change_pct": round(mean_diff_pct, 2)
                })
            else:
                # 2. Categorical Feature Frequency Shift
                ref_freq = ref_series.value_counts(normalize=True).to_dict()
                tar_freq = tar_series.value_counts(normalize=True).to_dict()
                
                # Calculate total variation distance
                all_keys = set(ref_freq.keys()).union(set(tar_freq.keys()))
                tvd = 0.5 * sum(abs(ref_freq.get(k, 0.0) - tar_freq.get(k, 0.0)) for k in all_keys)
                
                if tvd > 0.3:
                    status = "DRIFT DETECTED"
                    drifted_count += 1
                elif tvd > 0.15:
                    status = "WARNING"
                    warning_count += 1
                else:
                    status = "NORMAL"

                feature_reports.append({
                    "feature": col,
                    "type": "categorical",
                    "status": status,
                    "p_value": None,
                    "tvd": round(float(tvd), 4),
                    "reference_mean": None,
                    "target_mean": None,
                    "mean_change_pct": round(tvd * 100, 2)
                })

        has_drift = drifted_count > 0
        overall_status = "DRIFT DETECTED" if has_drift else ("WARNING" if warning_count > 0 else "NORMAL")
        
        report_data = {
            "id": generate_unique_id("drift"),
            "reference_dataset": reference_dataset_id,
            "target_dataset": target_dataset_id,
            "has_drift": has_drift,
            "drift_status": overall_status,
            "drifted_features_count": drifted_count,
            "warning_features_count": warning_count,
            "total_features_analyzed": len(feature_reports),
            "feature_reports": feature_reports,
            "timestamp": get_current_timestamp_iso()
        }
        
        # Save drift report to monitoring storage
        MonitoringService.save_drift_report(report_data)
        return report_data

# Feature sync: feature/monitoring-drift-retraining (PR #12)

# Feature sync: feature/monitoring-drift-retraining (PR #12)
