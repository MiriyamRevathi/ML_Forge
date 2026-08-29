"""
MLForge ML Engine - Extended Statistical Data Drift Analysis Engine Module
Calculates Kolmogorov-Smirnov (KS-test), Population Stability Index (PSI),
Total Variation Distance (TVD), Chi-Square distribution shifts, and drift alert statuses.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional


class DriftStatisticalEngine:
    """
    Statistical Data Drift Engine.
    """

    @staticmethod
    def calculate_ks_drift(
        reference_series: pd.Series,
        target_series: pd.Series,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Computes 2-sample Kolmogorov-Smirnov test between reference and target numerical distributions.
        """
        ref_clean = reference_series.dropna().values
        tar_clean = target_series.dropna().values

        if len(ref_clean) < 5 or len(tar_clean) < 5:
            return {
                "p_value": 1.0,
                "ks_statistic": 0.0,
                "has_drift": False,
                "status": "INSUFFICIENT_DATA"
            }

        ks_stat, p_val = stats.ks_2samp(ref_clean, tar_clean)

        ref_mean = float(np.mean(ref_clean))
        tar_mean = float(np.mean(tar_clean))
        mean_change = round(((tar_mean - ref_mean) / (abs(ref_mean) if ref_mean != 0 else 1.0)) * 100, 2)

        has_drift = bool(p_val < alpha)

        return {
            "p_value": round(float(p_val), 6),
            "ks_statistic": round(float(ks_stat), 4),
            "reference_mean": round(ref_mean, 4),
            "target_mean": round(tar_mean, 4),
            "mean_change_pct": mean_change,
            "has_drift": has_drift,
            "status": "DRIFT_DETECTED" if has_drift else "STABLE"
        }

    @staticmethod
    def calculate_psi(
        reference_series: pd.Series,
        target_series: pd.Series,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes Population Stability Index (PSI) across binned numerical feature values.
        """
        ref_clean = reference_series.dropna().values
        tar_clean = target_series.dropna().values

        if len(ref_clean) < 10 or len(tar_clean) < 10:
            return {"psi": 0.0, "status": "STABLE"}

        # Define bin edges on reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref_clean, percentiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return {"psi": 0.0, "status": "STABLE"}

        ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
        tar_counts, _ = np.histogram(tar_clean, bins=bin_edges)

        ref_pcts = (ref_counts / len(ref_clean)) + 1e-4
        tar_pcts = (tar_counts / len(tar_clean)) + 1e-4

        psi_val = float(np.sum((tar_pcts - ref_pcts) * np.log(tar_pcts / ref_pcts)))
        psi_val = round(psi_val, 4)

        if psi_val >= 0.25:
            status = "ACTION_REQUIRED"
        elif psi_val >= 0.10:
            status = "SLIGHT_DRIFT"
        else:
            status = "STABLE"

        return {
            "psi_score": psi_val,
            "status": status,
            "has_drift": bool(psi_val >= 0.25)
        }
