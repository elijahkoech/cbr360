# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# RAW_TARGETS_PATH = ROOT / "data" / "data.csv"
# RAW_KPIS_PATH = ROOT / "data" / "data.csv"

RAW_DATA_DIR =  ROOT / "data" / "raw"
PROCESSED_DATA_DIR =  ROOT / "data" / "processed"
MODELS_DIR =  ROOT / "models"