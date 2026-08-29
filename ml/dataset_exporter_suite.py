"""
MLForge ML Engine - Extended Dataset Exporter Suite Module
Exports DataFrames to CSV, JSON, formatted HTML tables, Markdown tables,
and summary data structures for client visualization or download.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from utils.files import validate_path_safety, save_json
from utils.helpers import make_json_serializable


class ExtendedDatasetExporterSuite:
    """
    Export utility for CSV, JSON, HTML, and Markdown format generation.
    """

    @staticmethod
    def export_to_csv(
        df: pd.DataFrame,
        filepath: Path,
        index: bool = False
    ) -> Path:
        safe_path = validate_path_safety(filepath, filepath.parent)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(safe_path, index=index)
        return safe_path

    @staticmethod
    def export_to_json(
        df: pd.DataFrame,
        filepath: Path,
        orient: str = "records"
    ) -> Path:
        records = df.to_dict(orient=orient)
        serializable_data = make_json_serializable(records)
        save_json(serializable_data, filepath)
        return filepath

    @staticmethod
    def export_to_html_table(
        df: pd.DataFrame,
        classes: str = "data-table",
        max_rows: int = 50
    ) -> str:
        display_df = df.head(max_rows)
        return display_df.to_html(classes=classes, index=False, border=0)

    @staticmethod
    def export_to_markdown_table(
        df: pd.DataFrame,
        max_rows: int = 20
    ) -> str:
        display_df = df.head(max_rows)
        return display_df.to_markdown(index=False)
