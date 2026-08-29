"""
MLForge ML Engine - Model Comparison & Multi-Model Benchmark Module
Ranks trained models across evaluation metrics, generates benchmark matrices,
and constructs visual comparison data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from services.model_service import ModelService
from ml.eda_visualizer import EDAVisualizer


class ModelComparisonEngine:
    """
    Multi-Model Benchmarking & Comparison Suite.
    """

    @staticmethod
    def compare_models(model_versions: List[str]) -> Dict[str, Any]:
        """
        Calculates metric ranking comparison across specified model versions.
        """
        models_meta = []
        for version in model_versions:
            meta = ModelService.get_model_metadata(version)
            if meta:
                models_meta.append(meta)

        if not models_meta:
            return {"ranked_models": [], "comparison_matrix": []}

        task_type = models_meta[0].get("task_type", "classification")

        # Sort models based on primary metric
        if task_type == "classification":
            primary_metric = "accuracy"
            models_meta.sort(key=lambda m: m.get("metrics", {}).get("accuracy", 0.0), reverse=True)
        else:
            primary_metric = "r2_score"
            models_meta.sort(key=lambda m: m.get("metrics", {}).get("r2_score", -999.0), reverse=True)

        ranked = []
        for rank, meta in enumerate(models_meta, start=1):
            metrics = meta.get("metrics", {})
            ranked.append({
                "rank": rank,
                "version": meta.get("version"),
                "model_name": meta.get("model_name"),
                "status": meta.get("status"),
                "primary_metric": primary_metric,
                "primary_value": metrics.get(primary_metric, 0.0),
                "metrics": metrics
            })

        return {
            "task_type": task_type,
            "primary_metric": primary_metric,
            "best_model_version": ranked[0]["version"] if ranked else None,
            "ranked_models": ranked
        }
