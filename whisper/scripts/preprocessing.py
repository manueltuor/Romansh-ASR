"""
ASR Data Preprocessing and Partitioning Pipeline.

This orchestrator script isolates raw data assets, runs a uniform text 
normalization pipeline (casing, punctuation stripping, and HTML cleaning) 
across dialect transcripts, constructs a 10% stratified verification split, 
and purges intermediate target manifests to prepare for training.
"""

import sys
import shutil
from pathlib import Path

script_dir = Path(__file__).resolve().parent
whisper_dir = script_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr.constants import CLEAN_DATA_ROOT as DATA_ROOT, RAW_DATA_ROOT, FOLDER_NAMES
from whisper_asr import preprocess_all_tsv_files, create_validation_splits, print_example

# Ensure the runtime workspace path directory trees exist safely on disk
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Clone the immutable source asset trees into our clean scratchpad workspace.
# dirs_exist_ok=True ensures idempotency if this script is rerun multiple times.
shutil.copytree(RAW_DATA_ROOT, DATA_ROOT, dirs_exist_ok=True)

print("Copy complete.")
print(f"Contents of {DATA_ROOT}:")
for item in sorted(DATA_ROOT.iterdir()):
    if item.is_dir():
        print(f"    {item.name}/")
    else:
        print(f"    {item.name}")

print("Cleaning HTML, punctuation and casing from all TSV files...")
# Strip raw corpus text columns of dirty web tags and normalize sentences
preprocess_all_tsv_files(data_root=DATA_ROOT, folder_names=FOLDER_NAMES)

print("\nSentences after cleaning:")
print_example()

print("Creating validation splits...")
# Slice out a clean 10% slice (`val_size=0.1`) across tracking categories 
# to establish local generalization metrics during training epochs.
create_validation_splits(data_root=DATA_ROOT, folder_names=FOLDER_NAMES, val_size=0.1)

# Remove all 'validated.tsv' files since we don't need them
for idiom_folder in FOLDER_NAMES:
    file_path = DATA_ROOT / idiom_folder / "validated.tsv"
    if file_path.exists():
        file_path.unlink()
        print(f"  Removed {file_path}")
    else:
        print(f"  {file_path} not found")

