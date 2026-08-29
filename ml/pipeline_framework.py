"""
MLForge ML Engine - Reusable DAG Pipeline Framework & Stage Controller Module
Defines PipelineStage, PipelineContext, PipelineCheckpoint, PipelineValidator,
PipelineMetrics, and PipelineSerializer for pipeline lifecycle management.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from utils.helpers import generate_unique_id, get_current_timestamp_iso


class PipelineStage:
    """
    Individual execution stage node in an ML Pipeline DAG.
    """
    def __init__(self, name: str, stage_type: str, action_func: Callable[..., Any]):
        self.stage_id = generate_unique_id("stage")
        self.name = name
        self.stage_type = stage_type
        self.action_func = action_func
        self.status = "PENDING"
        self.execution_time_seconds = 0.0
        self.output = None
        self.error_message = None

    def execute(self, context: 'PipelineContext') -> Any:
        start_t = time.time()
        self.status = "RUNNING"
        context.logger.info(f"Stage [{self.name}] starting execution...")
        try:
            self.output = self.action_func(context)
            self.status = "COMPLETED"
            self.execution_time_seconds = round(time.time() - start_t, 3)
            context.logger.info(f"Stage [{self.name}] completed in {self.execution_time_seconds}s.")
            return self.output
        except Exception as e:
            self.status = "FAILED"
            self.error_message = str(e)
            self.execution_time_seconds = round(time.time() - start_t, 3)
            context.logger.error(f"Stage [{self.name}] failed: {str(e)}")
            raise e


class PipelineContext:
    """
    Shared execution state context passed across DAG stages.
    """
    def __init__(self, pipeline_config: Dict[str, Any], logger: Any):
        self.config = pipeline_config
        self.logger = logger
        self.artifacts: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.stage_outputs: Dict[str, Any] = {}

    def set_artifact(self, key: str, value: Any):
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Optional[Any] = None) -> Any:
        return self.artifacts.get(key, default)

    def set_metric(self, key: str, value: Any):
        self.metrics[key] = value


class PipelineCheckpoint:
    """
    Pipeline state snapshot and checkpoint manager.
    """
    @staticmethod
    def create_checkpoint(context: PipelineContext, stage_name: str) -> Dict[str, Any]:
        return {
            "checkpoint_id": generate_unique_id("chk"),
            "stage_name": stage_name,
            "timestamp": get_current_timestamp_iso(),
            "artifacts_keys": list(context.artifacts.keys()),
            "metrics": context.metrics
        }
