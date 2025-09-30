# config.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent

# RAW_TARGETS_PATH = ROOT / "data" / "data.csv"
# RAW_KPIS_PATH = ROOT / "data" / "data.csv"

RAW_DATA_DIR =  ROOT / "data" / "raw"
PROCESSED_DATA_DIR =  ROOT / "data" / "processed"
MODELS_DIR =  ROOT / "models"

def setup_project_path():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
