import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw-data"
CLEAN_DATA_ROOT = DATA_ROOT / "clean-data"
PARQUET_DATA_ROOT = DATA_ROOT / "parquet-data"