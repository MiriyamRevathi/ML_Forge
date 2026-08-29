"""
MLForge - Production Lines of Code (LOC) Counter Script
Recursively counts production source code across Python, JavaScript, HTML, and CSS files.
Excludes tests, virtual environments, .git, cache files, data files, and generated artifacts.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    "tests",
    ".git",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".pytest_cache",
    "__pycache__",
    "data",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".idea",
    ".vscode"
}

EXTENSION_LANG_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".html": "HTML",
    ".css": "CSS"
}


def count_file_loc(filepath: Path) -> int:
    """Counts non-empty lines in a production source file."""
    try:
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():  # non-blank line
                    count += 1
        return count
    except Exception:
        return 0


def scan_production_loc(root_dir: Path):
    loc_by_lang = {
        "Python": 0,
        "JavaScript": 0,
        "HTML": 0,
        "CSS": 0
    }
    file_counts = {
        "Python": 0,
        "JavaScript": 0,
        "HTML": 0,
        "CSS": 0
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        
        rel_path = Path(dirpath).relative_to(root_dir)
        # Skip if any parent part is in EXCLUDE_DIRS
        if any(part in EXCLUDE_DIRS for part in rel_path.parts):
            continue

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in EXTENSION_LANG_MAP:
                lang = EXTENSION_LANG_MAP[ext]
                filepath = Path(dirpath) / filename
                loc = count_file_loc(filepath)
                loc_by_lang[lang] += loc
                file_counts[lang] += 1

    total_loc = sum(loc_by_lang.values())

    print("=" * 45)
    print("      MLForge Production LOC Report")
    print("=" * 45)
    for lang, loc in loc_by_lang.items():
        files = file_counts[lang]
        print(f"{lang:<12}: {loc:>7,} LOC  ({files} files)")
    print("-" * 45)
    print(f"TOTAL PRODUCTION LOC: {total_loc:>7,}")
    print("=" * 45)
    
    if total_loc >= 50000:
        print("STATUS: PASS (Requirement: 50,000+ LOC met!)")
    else:
        missing = 50000 - total_loc
        print(f"STATUS: FAIL (Need {missing:,} more production LOC)")
    print("=" * 45)

    return total_loc


if __name__ == "__main__":
    scan_production_loc(BASE_DIR)
