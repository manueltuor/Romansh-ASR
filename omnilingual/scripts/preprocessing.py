#!/usr/bin/env python3
"""
Convert Romansh TSV to Omnilingual Parquet with smaller fragments.
Run from the omnilingual-romansh directory.
"""

import os
import io
import sys
import pandas as pd
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm
from omnilingual_asr.utils import get_idiom_name_by_folder, normalize_romansh_text, get_language_code_by_folder
from omnilingual_asr.constants import FOLDER_NAMES, DATA_ROOT, PARQUET_DATA_ROOT

scripts_dir = Path(__file__).resolve().parent
submodule_root = scripts_dir.parent / 'omnilingual_asr'
sys.path.insert(0, str(submodule_root))

# ----------------------------------------------------------------------
# Configuration – adjust these!
# ----------------------------------------------------------------------
OUTPUT_ROOT = PARQUET_DATA_ROOT                # Output Parquet dataset location
BATCH_SIZE = 100                               # Number of rows per output file (smaller)
ROW_GROUP_SIZE = 50                            # Rows per row group (smaller)

# ----------------------------------------------------------------------
# Helper: compress audio to OGG
# ----------------------------------------------------------------------
def compress_audio_to_ogg(audio_array, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='ogg')
    return buffer.getvalue()

# ----------------------------------------------------------------------
# Import official conversion utility (from the copied repo)
# ----------------------------------------------------------------------
sys.path.insert(0, os.getcwd())                # add current dir to Python path
from omnilingual_asr.workflows.dataprep.audio_tools import binary_to_list_int8
print("✅ Using official binary_to_list_int8")

# ----------------------------------------------------------------------
# Write a batch to Parquet
# ----------------------------------------------------------------------
def write_batch(records, out_dir, part_idx):
    binary_array = pa.array([r['audio_bytes'] for r in records], type=pa.binary())
    audio_bytes_list = binary_to_list_int8(binary_array)

    table = pa.Table.from_pydict({
        'text': [r['text'] for r in records],
        'audio_bytes': audio_bytes_list,
        'audio_size': [r['audio_size'] for r in records],
    })

    out_path = os.path.join(out_dir, f"part-{part_idx}.parquet")
    pq.write_table(table, out_path, row_group_size=ROW_GROUP_SIZE)
    print(f"  Wrote {len(records)} rows to {out_path}")

# ----------------------------------------------------------------------
# Process one split (train/validation) for one idiom
# ----------------------------------------------------------------------
def process_split(idiom_folder, split_name):
    tsv_path = os.path.join(DATA_ROOT, idiom_folder, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        print(f"⚠️ {tsv_path} not found, skipping.")
        return

    clips_dir = os.path.join(DATA_ROOT, idiom_folder, "clips")
    df = pd.read_csv(tsv_path, sep='\t')

    # Use human‑readable corpus name for directory structure
    corpus_name = get_idiom_name_by_folder(idiom_folder)
    language_code = get_language_code_by_folder(idiom_folder)
    out_dir = os.path.join(OUTPUT_ROOT, f"corpus={corpus_name}", f"split={split_name}", f"language={language_code}")
    os.makedirs(out_dir, exist_ok=True)

    batch_records = []
    part_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{corpus_name}/{split_name}"):
        audio_path = os.path.join(clips_dir, row['path'])
        try:
            audio, sr = sf.read(audio_path)

            ogg_bytes = compress_audio_to_ogg(audio, sr)
            audio_size = len(audio)

            batch_records.append({
                'text': normalize_romansh_text(row['sentence']),
                'audio_bytes': ogg_bytes,
                'audio_size': audio_size,
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

        if len(batch_records) >= BATCH_SIZE:
            write_batch(batch_records, out_dir, part_idx)
            part_idx += 1
            batch_records = []

    if batch_records:
        write_batch(batch_records, out_dir, part_idx)

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    print("="*60)
    print("Converting Romansh data to Omnilingual Parquet (small fragments)")
    print("="*60)
    print(f"Data root: {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Batch size: {BATCH_SIZE}, Row group size: {ROW_GROUP_SIZE}")
    print("="*60)

    for idiom_folder in FOLDER_NAMES:
        corpus_name = get_idiom_name_by_folder(idiom_folder)
        print(f"\n📂 Processing idiom: {corpus_name} (folder: {idiom_folder})")
        for split in ['train', 'validation', 'test']: 
            print(f"  Split: {split}")
            process_split(idiom_folder, split)

    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()