"""
MLForge - Dataset Management Service Module
Provides dataset file access, metadata tracking, CSV preview generation,
file listing, dataset registration, validation audit, and EDA report generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from config import DATASET_DIR, SAMPLE_DIR
from utils.files import load_json, save_json, delete_file, get_file_size_formatted, list_files_in_dir
from utils.helpers import get_current_timestamp_iso, generate_unique_id
from ml.dataset_loader import DatasetLoader
from ml.validation import DataValidator
from ml.exploration import ExploratoryDataAnalysis


class DatasetService:
    """
    Service responsible for dataset operations, validation audits, and EDA chart generation.
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
    def load_dataset_dataframe(dataset_id: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Reads dataset CSV file into a Pandas DataFrame using DatasetLoader.
        """
        csv_path = DatasetService.get_dataset_csv_path(dataset_id)
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV for ID '{dataset_id}' not found.")
            
        return DatasetLoader.load_csv(csv_path, max_rows=max_rows)

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
        
        df = pd.read_csv(filepath)
        df.to_csv(dest_path, index=False)
        
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)
        
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
    def validate_dataset(dataset_id: str) -> Dict[str, Any]:
        """
        Executes data quality checks on dataset and returns validation report.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            raise FileNotFoundError(f"Dataset ID '{dataset_id}' not found.")
            
        df = DatasetService.load_dataset_dataframe(dataset_id)
        target_col = meta.get("target_column")
        
        audit_results = DataValidator.validate_dataset(df, target_column=target_col)
        audit_results["dataset"] = meta
        return audit_results

    @staticmethod
    def run_eda(dataset_id: str) -> Dict[str, Any]:
        """
        Generates comprehensive Exploratory Data Analysis report with statistical summaries and base64 Matplotlib charts.
        """
        meta = DatasetService.get_dataset_metadata(dataset_id)
        if not meta:
            raise FileNotFoundError(f"Dataset ID '{dataset_id}' not found.")
            
        df = DatasetService.load_dataset_dataframe(dataset_id)
        target_col = meta.get("target_column", "")
        
        num_stats = ExploratoryDataAnalysis.get_numerical_statistics(df)
        cat_stats = ExploratoryDataAnalysis.get_categorical_statistics(df)
        corr_matrix = ExploratoryDataAnalysis.get_correlation_matrix(df)
        
        charts = {
            "distribution": ExploratoryDataAnalysis.create_distribution_chart(df),
            "boxplot": ExploratoryDataAnalysis.create_box_plot_chart(df),
            "correlation": ExploratoryDataAnalysis.create_correlation_heatmap(df),
            "missing_values": ExploratoryDataAnalysis.create_missing_values_chart(df),
            "target": ExploratoryDataAnalysis.create_target_distribution_chart(df, target_col)
        }
        
        return {
            "dataset": meta,
            "numerical_statistics": num_stats,
            "categorical_statistics": cat_stats,
            "correlation_matrix": corr_matrix,
            "charts": charts
        }

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
