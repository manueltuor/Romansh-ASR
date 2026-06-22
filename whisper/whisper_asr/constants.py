import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
WHISPER_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
CLEAN_DATA_ROOT = DATA_ROOT / "clean-data"
RAW_DATA_ROOT = DATA_ROOT / "raw-data"
MODELS_ROOT = WHISPER_ROOT / "models"
FOLDER_NAMES = [folder for folder in os.listdir(RAW_DATA_ROOT) if "cc" in folder]
SPLITS = ["train", "validated", "test", "validation"]
SPLIT_FILES = ["train.tsv", "validated.tsv", "test.tsv", "validation.tsv"]
RANDOM_SEED = 42