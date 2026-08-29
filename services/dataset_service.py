"""
MLForge - Dataset Management Service Module
Provides dataset file access, metadata tracking, CSV preview generation,
file listing, dataset registration, and dataset deletion operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from config import DATASET_DIR, SAMPLE_DIR
from utils.files import load_json, save_json, delete_file, get_file_size_formatted, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id


class DatasetService:
    """
    Service responsible for dataset operations and metadata management.
    """
    
    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        """
        Lists all registered datasets with metadata.
        """
        meta_files = list_files_in_dir(DATASET_DIR, extension="json")
        datasets = []
        
        for meta_file in meta_files:
            if meta_file.name.endswith("_meta.json"):
                try:
                    metadata = load_json(meta_file)
                    datasets.append(metadata)
                except Exception:
                    continue
                    
        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)

    @staticmethod
    def get_dataset_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata JSON for a dataset by ID.
        """
        meta_path = DATASET_DIR / f"{dataset_id}_meta.json"
        if meta_path.exists():
            return load_json(meta_path)
            
        # Try direct filename matching
        meta_files = list_files_in_dir(DATASET_DIR, extension="json")
        for mf in meta_files:
            meta = load_json(mf)
            if meta.get("id") == dataset_id or meta.get("filename") == dataset_id:
                return meta
                
        return None

    @staticmethod
    def get_dataset_csv_path(dataset_id: str) -> Optional[Path]:
        """
        Returns the absolute file path to the CSV file of a given dataset ID.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if meta and "filename" in meta:
            csv_path = DATASET_DIR / meta["filename"]
            if csv_path.exists():
                return csv_path
                
        # Try matching direct filename
        direct_path = DATASET_DIR / dataset_id
        if direct_path.exists() and direct_path.suffix == ".csv":
            return direct_path
            
        direct_path_csv = DATASET_DIR / f"{dataset_id}.csv"
        if direct_path_csv.exists():
            return direct_path_csv
            
        return None

    @staticmethod
    def load_dataset_dataframe(dataset_id: str) -> pd.DataFrame:
        """
        Reads dataset CSV file into a Pandas DataFrame.
        """
        csv_path = DatasetService.get_dataset_csv_path(dataset_id)
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV for ID '{dataset_id}' not found.")
            
        return pd.read_csv(csv_path)

    @staticmethod
    def register_dataset(
        filepath: Path,
        custom_name: Optional[str] = None,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registers a new dataset CSV file into storage and creates metadata JSON.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Source dataset file '{filepath}' does not exist.")
            
        dataset_id = generate_unique_id("ds")
        dest_filename = f"{dataset_id}.csv"
        dest_path = DATASET_DIR / dest_filename
        
        # Read df to gather statistics
        df = pd.read_csv(filepath)
        
        # Copy CSV to dataset dir
        df.to_csv(dest_path, index=False)
        
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        metadata = {
            "id": dataset_id,
            "filename": dest_filename,
            "original_filename": filepath.name,
            "name": custom_name or filepath.stem.replace("_", " ").title(),
            "target_column": target_column or (df.columns[-1] if len(df.columns) > 0 else ""),
            "task_type": task_type or "classification",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_values_count": int(df.isna().sum().sum()),
            "file_size": get_file_size_formatted(dest_path),
            "created_at": get_current_timestamp_iso(),
            "is_sample": False
        }
        
        meta_path = DATASET_DIR / f"{dataset_id}_meta.json"
        save_json(metadata, meta_path)
        return metadata

    @staticmethod
    def delete_dataset(dataset_id: str) -> bool:
        """
        Deletes a dataset CSV file and its associated metadata JSON.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if meta:
            filename = meta.get("filename")
            if filename:
                delete_file(DATASET_DIR / filename)
            delete_file(DATASET_DIR / f"{dataset_id}_meta.json")
            return True
        return False
