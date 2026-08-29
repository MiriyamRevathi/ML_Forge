"""
MLForge ML Engine - Model Registry State Machine & Lifecycle Module
Manages state transitions (TRAINED, VALIDATED, STAGING, PRODUCTION, ARCHIVED),
promotions, demotions, and deployment validations.
"""

from typing import Dict, List, Any, Optional
from config import STATUS_TRAINED, STATUS_VALIDATED, STATUS_STAGING, STATUS_PRODUCTION, STATUS_ARCHIVED, REGISTRY_STATUSES
from services.model_service import ModelService


class ModelRegistryStateMachine:
    """
    Model Registry lifecycle state machine manager.
    """
    
    VALID_TRANSITIONS = {
        STATUS_TRAINED: [STATUS_VALIDATED, STATUS_STAGING, STATUS_ARCHIVED],
        STATUS_VALIDATED: [STATUS_STAGING, STATUS_PRODUCTION, STATUS_ARCHIVED],
        STATUS_STAGING: [STATUS_PRODUCTION, STATUS_ARCHIVED],
        STATUS_PRODUCTION: [STATUS_STAGING, STATUS_ARCHIVED],
        STATUS_ARCHIVED: [STATUS_STAGING, STATUS_TRAINED]
    }

    @staticmethod
    def can_transition(current_status: str, target_status: str) -> bool:
        """
        Validates if transition from current_status to target_status is permitted.
        """
        allowed = ModelRegistryStateMachine.VALID_TRANSITIONS.get(current_status, REGISTRY_STATUSES)
        return target_status in allowed

    @staticmethod
    def promote_to_production(model_version: str) -> Dict[str, Any]:
        """
        Promotes a model version to PRODUCTION status and demotes previous PRODUCTION model.
        """
        updated = ModelService.update_model_status(model_version, STATUS_PRODUCTION)
        if not updated:
            raise ValueError(f"Model version '{model_version}' not found in registry.")
        return updated

    @staticmethod
    def archive_model(model_version: str) -> Dict[str, Any]:
        """
        Archives a model version.
        """
        updated = ModelService.update_model_status(model_version, STATUS_ARCHIVED)
        if not updated:
            raise ValueError(f"Model version '{model_version}' not found in registry.")
        return updated

# Feature sync: feature/model-registry-state-machine (PR #10)

# Feature sync: feature/model-registry-state-machine (PR #10)
