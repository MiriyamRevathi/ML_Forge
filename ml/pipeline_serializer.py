"""
MLForge ML Engine - Pipeline DAG Serializer & Export/Import Module
Handles JSON serialization, deserialization, cloning, exporting, and importing of pipeline DAGs.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from utils.files import save_json, load_json, validate_path_safety
from utils.helpers import generate_unique_id, get_current_timestamp_iso


class PipelineSerializer:
    """
    Pipeline DAG specification serialization utility.
    """

    @staticmethod
    def export_pipeline_spec_to_file(spec: Dict[str, Any], filepath: Path) -> Path:
        """
        Exports pipeline spec payload to JSON file safely.
        """
        safe_path = validate_path_safety(filepath, filepath.parent)
        save_json(spec, safe_path)
        return safe_path

    @staticmethod
    def import_pipeline_spec_from_file(filepath: Path) -> Dict[str, Any]:
        """
        Imports and validates pipeline spec JSON from disk.
        """
        safe_path = validate_path_safety(filepath, filepath.parent)
        return load_json(safe_path)

    @staticmethod
    def clone_pipeline_spec(original_spec: Dict[str, Any], new_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a copy/clone of a pipeline DAG specification with a new unique ID.
        """
        cloned = json.loads(json.dumps(original_spec))
        cloned["id"] = generate_unique_id("pipe")
        cloned["name"] = new_name or f"Copy of {original_spec.get('name', 'Pipeline')}"
        cloned["created_at"] = get_current_timestamp_iso()
        return cloned
