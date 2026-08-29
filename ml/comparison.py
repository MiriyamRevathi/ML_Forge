"""
MLForge ML Engine - Model Comparison Module
Aggregates metrics across multiple trained models, sorts by target performance metric,
identifies best performing model, and generates Matplotlib comparison bar charts.
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional


class ModelComparer:
    """
    Multi-model comparison and benchmark chart generator.
    """
    
    @staticmethod
    def compare_models(
        models_metadata: List[Dict[str, Any]],
        primary_metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compares multiple trained model metadata records and highlights best model.
        """
        if not models_metadata:
            return {"models": [], "best_model": None, "comparison_chart": None}
            
        # Determine primary metric based on first model task type
        first_task = models_metadata[0].get("task_type", "classification")
        metric_key = primary_metric or ("accuracy" if first_task == "classification" else "r2_score")
        
        comparison_list = []
        for m in models_metadata:
            metrics = m.get("metrics", {})
            val = metrics.get(metric_key, 0.0) if isinstance(metrics, dict) else 0.0
            comparison_list.append({
                "version": m.get("version"),
                "name": m.get("name"),
                "task_type": m.get("task_type"),
                "status": m.get("status"),
                "primary_metric_name": metric_key,
                "primary_metric_value": val if val is not None else 0.0,
                "metrics": metrics
            })
            
        # Sort descending by primary metric
        sorted_models = sorted(comparison_list, key=lambda x: x["primary_metric_value"], reverse=True)
        best_model = sorted_models[0] if sorted_models else None
        
        # Generate comparison bar chart
        chart_base64 = ModelComparer.generate_comparison_chart(sorted_models, metric_key)

        return {
            "models": sorted_models,
            "best_model": best_model,
            "primary_metric": metric_key,
            "comparison_chart": chart_base64
        }

    @staticmethod
    def generate_comparison_chart(
        sorted_models: List[Dict[str, Any]],
        metric_name: str
    ) -> Optional[str]:
        """
        Generates Matplotlib bar chart comparing model metrics.
        """
        if not sorted_models:
            return None
            
        labels = [m["name"][:15] + " (" + m["version"] + ")" for m in sorted_models[:8]]
        values = [m["primary_metric_value"] for m in sorted_models[:8]]
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#161e2e')
        ax.set_facecolor('#111827')
        
        bars = ax.barh(labels[::-1], values[::-1], color='#3b82f6', alpha=0.85)
        if bars:
            bars[0].set_color('#10b981')  # Highlight top model bar in green
            
        ax.set_title(f"Model Comparison Benchmark ({metric_name.replace('_', ' ').title()})", color='#f9fafb', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#9ca3af', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.2)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
