from pathlib import Path

# Project Root is determined dynamically based on the location of this file
# src/config.py is located at <project_root>/src/config.py
# Therefore, the project root is its parent's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifact Paths
MODELS_DIR = PROJECT_ROOT / "models"
TUNING_LOGS_DIR = PROJECT_ROOT / "tuning_logs"

# Ensure essential directories exist when config is loaded
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_LOGS_DIR.mkdir(parents=True, exist_ok=True)
