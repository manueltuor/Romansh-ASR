import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
WHISPER_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
CLEAN_DATA_ROOT = DATA_ROOT / "clean-data"
RAW_DATA_ROOT = DATA_ROOT / "raw-data"
MODELS_ROOT = WHISPER_ROOT / "models"
FOLDER_NAMES = [
  "rm-cc-2021-05-28",
  "rmputer-cc-2021-06-11",
  "rmsursilv-cc-2021-05-28",
  "rmsursiv-cc-2021-12-23",
  "rmsutsilv-cc-2022-05-18",
  "rmvallader-cc-2021-05-28"
]
SPLITS = ["train", "validated", "test", "validation"]
SPLIT_FILES = ["train.tsv", "validated.tsv", "test.tsv", "validation.tsv"]
RANDOM_SEED = 42