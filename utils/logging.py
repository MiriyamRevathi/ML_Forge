"""
MLForge - Platform Logging & Execution Logger Module
Provides real-time pipeline execution logging, structured formatting,
in-memory log buffers, and persistent log file writing.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import LOG_DIR

# Set up global base logger
logger = logging.getLogger("mlforge")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


class PipelineExecutionLogger:
    """
    In-memory and persistent logger designed specifically for tracking step-by-step ML Pipeline executions.
    """
    
    def __init__(self, pipeline_run_id: str, log_to_file: bool = True):
        self.pipeline_run_id = pipeline_run_id
        self.logs: List[Dict[str, Any]] = []
        self.log_to_file = log_to_file
        self.log_filepath = LOG_DIR / f"pipeline_run_{pipeline_run_id}.log"
        
        self.info(f"Initializing pipeline execution log for run ID: {pipeline_run_id}")

    def _add_entry(self, level: str, message: str) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        iso_timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "iso_timestamp": iso_timestamp,
            "level": level,
            "message": message,
            "formatted": f"[{timestamp}] [{level}] {message}"
        }
        
        self.logs.append(entry)
        
        # Log to global system stdout logger as well
        if level == "ERROR":
            logger.error(f"[Run {self.pipeline_run_id}] {message}")
        elif level == "WARNING":
            logger.warning(f"[Run {self.pipeline_run_id}] {message}")
        else:
            logger.info(f"[Run {self.pipeline_run_id}] {message}")
            
        if self.log_to_file:
            self._write_to_file(entry["formatted"])
            
        return entry

    def info(self, message: str):
        """Logs an INFO level message."""
        return self._add_entry("INFO", message)

    def warning(self, message: str):
        """Logs a WARNING level message."""
        return self._add_entry("WARNING", message)

    def error(self, message: str):
        """Logs an ERROR level message."""
        return self._add_entry("ERROR", message)

    def _write_to_file(self, line: str):
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error(f"Failed to append to pipeline log file {self.log_filepath}: {e}")

    def get_formatted_logs(self) -> List[str]:
        """Returns list of formatted string log lines."""
        return [entry["formatted"] for entry in self.logs]

    def get_structured_logs(self) -> List[Dict[str, Any]]:
        """Returns list of structured log dictionaries."""
        return self.logs


def get_system_logger() -> logging.Logger:
    """Returns the primary MLForge system logger instance."""
    return logger
