import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from typing import List, Optional

from .constants import DATA_ROOT, FOLDER_NAMES

def clean_html(text: Optional[str]) -> str:
    """
    Clean HTML tags and special characters from a text string.
    Handles None or non-string inputs by returning an empty string.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?p\s*/?>', ' ', text, flags=re.IGNORECASE)

    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_tsv_file(tsv_path: str) -> int:
    """
    Clean the 'sentence' column of a TSV file in-place.
    Returns the number of rows processed, or 0 if an error occurs.
    """
    if not os.path.exists(tsv_path):
        print(f"Warning: File not found - {tsv_path}")
        return 0

    df = pd.read_csv(tsv_path, sep='\t')
    if 'sentence' not in df.columns:
        print(f"Warning: No 'sentence' column found in {tsv_path}")
        return 0

    df['sentence'] = df['sentence'].apply(clean_html)
    df.to_csv(tsv_path, sep='\t', index=False)
    return len(df)

def preprocess_all_tsv_files(data_root: str = DATA_ROOT, folder_names: List[str] = FOLDER_NAMES) -> None:
    """
    Find all TSV files in the specified folders and clean their HTML.
    """
    for folder in folder_names:
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found - {folder_path}")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.tsv'):
                tsv_path = os.path.join(folder_path, file_name)
                print(f"Cleaning: {tsv_path}")
                num_rows = preprocess_tsv_file(tsv_path)
                print(f"  → Cleaned {num_rows} rows.")

def create_validation_splits(
    data_root: str = DATA_ROOT,
    folder_names: List[str] = FOLDER_NAMES,
    val_size: float = 0.1,
    random_state: int = 42
) -> None:
    """
    For each idiom, split train.tsv into a new train.tsv (90%) and a validation.tsv (10%).
    If validation.tsv already exists, skip that idiom.
    """
    for idiom_folder in folder_names:
        idiom_path = os.path.join(data_root, idiom_folder)
        train_path = os.path.join(idiom_path, "train.tsv")
        val_path = os.path.join(idiom_path, "validation.tsv")

        if not os.path.exists(train_path):
            print(f"Warning: train.tsv not found in {idiom_path}, skipping.")
            continue

        if os.path.exists(val_path):
            print(f"Validation split already exists for {idiom_folder}, skipping.")
            continue

        print(f"Creating validation split for {idiom_folder}...")
        train_df = pd.read_csv(train_path, sep='\t')
        train_data, val_data = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state
        )

        train_data.to_csv(train_path, sep='\t', index=False)
        val_data.to_csv(val_path, sep='\t', index=False)

    print("\n" + "=" * 60)
    print("All validation splits created successfully!")
    print("=" * 60)

def print_example(data_root: str = DATA_ROOT, folder_name: str = FOLDER_NAMES[0], n: int = 3) -> None:
    """
    Print the first few cleaned sentences from a given idiom's train.tsv.
    Useful for verifying the cleaning process.
    """
    example_path = os.path.join(data_root, folder_name, "train.tsv")
    if not os.path.exists(example_path):
        print(f"File not found: {example_path}")
        return

    example_df = pd.read_csv(example_path, sep='\t')
    for i in range(min(n, len(example_df))):
        print(f" {i+1}: {example_df['sentence'].iloc[i]}")