import sys
import shutil
from pathlib import Path

notebook_dir = Path.cwd()
whisper_dir = notebook_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr.constants import CLEAN_DATA_ROOT as DATA_ROOT, RAW_DATA_ROOT, FOLDER_NAMES
from whisper_asr import preprocess_all_tsv_files, create_validation_splits, print_example

DATA_ROOT.mkdir(parents=True, exist_ok=True)

shutil.copytree(RAW_DATA_ROOT, DATA_ROOT, dirs_exist_ok=True)

print("Copy complete.")
print(f"Contents of {DATA_ROOT}:")
for item in sorted(DATA_ROOT.iterdir()):
    if item.is_dir():
        print(f"    {item.name}/")
    else:
        print(f"    {item.name}")

print("Cleaning HTML, punctuation and casing from all TSV files...")
preprocess_all_tsv_files(data_root=DATA_ROOT, folder_names=FOLDER_NAMES)

print("\nSentences after cleaning:")
print_example()

print("Creating validation splits...")
create_validation_splits(data_root=DATA_ROOT, folder_names=FOLDER_NAMES, val_size=0.1)

for idiom_folder in FOLDER_NAMES:
    file_path = DATA_ROOT / idiom_folder / "validated.tsv"
    if file_path.exists():
        file_path.unlink()
        print(f"  Removed {file_path}")
    else:
        print(f"  {file_path} not found")

