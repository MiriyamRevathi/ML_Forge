"""
MLForge ML Engine - Automated Retraining Module
Triggers model retraining pipelines on new or drifted datasets, evaluates candidate model
against current production baseline, and automatically promotes superior models.
"""

from typing import Dict, List, Any, Optional
from services.model_service import ModelService
from services.pipeline_service import PipelineService
from ml.pipeline import PipelineEngine
from ml.registry import ModelRegistryStateMachine


class ModelRetrainer:
    """
    Automated model retraining & promotion engine.
    """

    @staticmethod
    def retrain_model(
        pipeline_id: str,
        target_dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes retraining pipeline and evaluates if new model version should be promoted to PRODUCTION.
        """
        pipeline_cfg = PipelineService.get_pipeline(pipeline_id)
        if not pipeline_cfg:
            raise FileNotFoundError(f"Pipeline configuration ID '{pipeline_id}' not found.")
            
        cfg_override = dict(pipeline_cfg)
        if target_dataset_id:
            cfg_override["dataset_id"] = target_dataset_id
            
        cfg_override["name"] = f"Retrained - {cfg_override.get('name')}"

        # 1. Execute retraining pipeline
        engine = PipelineEngine(cfg_override)
        run_summary = engine.execute()
        
        candidate_version = run_summary["model_version"]
        candidate_meta = ModelService.get_model_metadata(candidate_version)
        candidate_metrics = run_summary["metrics"]

        # 2. Get current PRODUCTION model for comparison
        task_type = candidate_meta.get("task_type")
        prod_model = ModelService.get_production_model(task_type=task_type)

        promoted = False
        promotion_reason = ""

        if not prod_model:
            # No production model exists; auto-promote candidate
            ModelRegistryStateMachine.promote_to_production(candidate_version)
            promoted = True
            promotion_reason = "Initial production deployment (no prior production model)."
        else:
            # Compare candidate metric against production metric
            metric_key = "accuracy" if task_type == "classification" else "r2_score"
            cand_score = candidate_metrics.get(metric_key, 0.0)
            prod_score = prod_model.get("metrics", {}).get(metric_key, 0.0)
            
            if cand_score > prod_score:
                ModelRegistryStateMachine.promote_to_production(candidate_version)
                promoted = True
                promotion_reason = f"Candidate score ({cand_score}) surpassed current Production baseline ({prod_score})."
            else:
                promotion_reason = f"Candidate score ({cand_score}) did not surpass current Production baseline ({prod_score}). Model kept in STAGING."

        return {
            "retrain_run": run_summary,
            "candidate_version": candidate_version,
            "promoted_to_production": promoted,
            "promotion_reason": promotion_reason
        }
