"""
MLForge - General Helper Functions Module
Provides formatting, date parsing, unique ID generation, and scalar transformation tools.
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Union


def generate_unique_id(prefix: str = "item") -> str:
    """
    Generates a unique timestamped identifier string.
    Example: dataset_20260829_a1b2c3d4
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{short_uuid}"


def get_current_timestamp_iso() -> str:
    """
    Returns current time formatted in standard ISO-8601 string.
    """
    return datetime.now().isoformat()


def get_current_timestamp_readable() -> str:
    """
    Returns human-friendly formatted current timestamp string.
    """
    return datetime.now().strftime("%b %d, %Y - %I:%M:%S %p")


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively converts numpy primitives, datetimes, and complex objects
    into standard Python JSON-serializable types.
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, pd.Series):
        return make_json_serializable(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return make_json_serializable(obj.to_dict(orient="records"))
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj


def format_duration(seconds: float) -> str:
    """
    Formats a execution duration in seconds into a human-readable string.
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        rem_seconds = seconds % 60
        return f"{minutes}m {rem_seconds:.1f}s"
