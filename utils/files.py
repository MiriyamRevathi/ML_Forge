"""
MLForge - Safe File Systems Utility Module
Provides robust, secure path resolution, file input/output operations,
JSON handling, joblib object persistence, and directory auditing.
"""

import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import werkzeug.utils
from config import BASE_DIR, DATA_DIR, ALLOWED_EXTENSIONS


class FileSystemError(Exception):
    """Custom exception raised for invalid file operations or path security violations."""
    pass


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes user-provided filename using secure_filename and removes path traversal vectors.
    """
    cleaned = werkzeug.utils.secure_filename(filename)
    if not cleaned:
        raise FileSystemError("Filename is empty or invalid after security sanitization.")
    return cleaned


def validate_path_safety(target_path: Path, base_dir: Path = DATA_DIR) -> Path:
    """
    Ensures that target_path resides within the permitted base_dir to prevent Path Traversal attacks.
    """
    resolved_target = target_path.resolve()
    resolved_base = base_dir.resolve()
    
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise FileSystemError(f"Security Alert: Path '{target_path}' attempts path traversal outside '{base_dir}'.")
    
    return resolved_target


def is_allowed_extension(filename: str) -> bool:
    """
    Checks if a given file has a permitted extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_json(data: Union[Dict, List], filepath: Path, indent: int = 4) -> bool:
    """
    Safely writes data to a JSON file.
    """
    try:
        safe_path = validate_path_safety(filepath, BASE_DIR)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first then rename to ensure atomic write operation
        temp_path = safe_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        
        temp_path.replace(safe_path)
        return True
    except Exception as e:
        raise FileSystemError(f"Failed to write JSON file at '{filepath}': {str(e)}")


def load_json(filepath: Path, default: Optional[Any] = None) -> Union[Dict, List, Any]:
    """
    Safely reads JSON file content. Returns default if file does not exist.
    """
    try:
        safe_path = validate_path_safety(filepath, BASE_DIR)
        if not safe_path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(f"JSON file not found: '{filepath}'")
            
        with open(safe_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if default is not None:
            return default
        raise
    except Exception as e:
        raise FileSystemError(f"Failed to read JSON file at '{filepath}': {str(e)}")


def save_joblib(obj: Any, filepath: Path) -> bool:
    """
    Serializes a Python object (e.g. scikit-learn model, transformer) to disk using joblib.
    """
    try:
        safe_path = validate_path_safety(filepath, BASE_DIR)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, safe_path)
        return True
    except Exception as e:
        raise FileSystemError(f"Failed to save binary artifact at '{filepath}': {str(e)}")


def load_joblib(filepath: Path) -> Any:
    """
    Loads a serialized joblib object from disk.
    """
    try:
        safe_path = validate_path_safety(filepath, BASE_DIR)
        if not safe_path.exists():
            raise FileNotFoundError(f"Binary artifact not found: '{filepath}'")
        return joblib.load(safe_path)
    except Exception as e:
        raise FileSystemError(f"Failed to load binary artifact at '{filepath}': {str(e)}")


def list_files_in_dir(directory: Path, extension: Optional[str] = None) -> List[Path]:
    """
    Lists all files in a target directory, optionally filtered by extension.
    """
    safe_dir = validate_path_safety(directory, BASE_DIR)
    if not safe_dir.exists() or not safe_dir.is_dir():
        return []
        
    files = [f for f in safe_dir.iterdir() if f.is_file()]
    if extension:
        ext = extension.lower() if extension.startswith('.') else f".{extension.lower()}"
        files = [f for f in files if f.suffix.lower() == ext]
        
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def delete_file(filepath: Path) -> bool:
    """
    Safely deletes a file from disk if it exists.
    """
    try:
        safe_path = validate_path_safety(filepath, BASE_DIR)
        if safe_path.exists():
            safe_path.unlink()
            return True
        return False
    except Exception as e:
        raise FileSystemError(f"Failed to delete file '{filepath}': {str(e)}")


def get_file_size_formatted(filepath: Path) -> str:
    """
    Returns a human-readable file size representation (e.g., 1.4 MB, 512 KB).
    """
    safe_path = validate_path_safety(filepath, BASE_DIR)
    if not safe_path.exists():
        return "0 B"
        
    size_bytes = safe_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
