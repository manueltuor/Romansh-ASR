"""
Convert Romansh TSV to Omnilingual Parquet with smaller fragments.
"""

import os
import pandas as pd
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm
from omnilingual_asr.utils import get_idiom_name_by_folder, normalize_romansh_text, get_language_code_by_folder
from omnilingual_asr.constants import FOLDER_NAMES, DATA_ROOT, PARQUET_DATA_ROOT
from omnilingual_asr.data import compress_audio_to_ogg, binary_to_list_int8

SCRIPTS_DIR = Path(__file__).resolve().parent          # scripts/
ROOT_DIR = SCRIPTS_DIR.parent                      # omnilingual/
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"         # submodule root (contains workflows/)

OUTPUT_ROOT = PARQUET_DATA_ROOT                # Output Parquet dataset location
BATCH_SIZE = 100                               # Number of rows per output file
ROW_GROUP_SIZE = 50                            # Rows per row group

def write_batch(records, out_dir, part_idx):
    """
    Constructs a PyArrow Table from a batch of processed records and writes 
    it out as a single Parquet part file.

    Args:
        records (list[dict]): List of dictionaries containing text, compressed bytes, and audio sizes.
        out_dir (str): Destination directory path where the Parquet file will be stored.
        part_idx (int): Chronological index sequence for numbering the output part files.
    """
    # Group raw audio byte elements into a PyArrow binary array representation
    binary_array = pa.array([r['audio_bytes'] for r in records], type=pa.binary())
    # Cast binary data to an explicit list of signed 8-bit integers (int8) required by the pipeline schema
    audio_bytes_list = binary_to_list_int8(binary_array)

    # Pack the tabular features safely into an immutable PyArrow Table structure
    table = pa.Table.from_pydict({
        'text': [r['text'] for r in records],
        'audio_bytes': audio_bytes_list,
        'audio_size': [r['audio_size'] for r in records],
    })

    # Generate a clean file destination and commit the table to disk with configured row group boundaries
    out_path = os.path.join(out_dir, f"part-{part_idx}.parquet")
    pq.write_table(table, out_path, row_group_size=ROW_GROUP_SIZE)
    print(f"  Wrote {len(records)} rows to {out_path}")
  
def process_split(idiom_folder, split_name):
    """
    Parses an idioms split's manifest TSV, reads and compresses the referenced 
    audio files on-the-fly, and serializes the data into a Hive-partitioned Parquet layout.

    Args:
        idiom_folder (str): Raw dataset source subdirectory containing transcripts and clips.
        split_name (str): The data split to be processed (e.g., 'train', 'validation', 'test').
    """
    # Resolve the manifest source location and skip execution gracefully if missing
    tsv_path = os.path.join(DATA_ROOT, idiom_folder, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        print(f"{tsv_path} not found, skipping.")
        return

    # Establish the root directory path for raw audio clips
    clips_dir = os.path.join(DATA_ROOT, idiom_folder, "clips")
    df = pd.read_csv(tsv_path, sep='\t')

    # Query mapping functions to extract clean metadata properties for folder routing
    corpus_name = get_idiom_name_by_folder(idiom_folder)
    language_code = get_language_code_by_folder(idiom_folder)

    # Omnilingual fine-tuning workflows require the strict keyword 'dev' instead of 'validation'
    if split_name == "validation":
        split_name_parquet = "dev"
    else:
        split_name_parquet = split_name

    # Construct standard Hive-style partition directories to enable automated parquet schema discoveries
    out_dir = os.path.join(OUTPUT_ROOT, f"corpus={corpus_name}", f"split={split_name_parquet}", f"language={language_code}")
    os.makedirs(out_dir, exist_ok=True)

    # Initialize micro-batch buffers to regulate server memory usage during streaming
    batch_records = []
    part_idx = 0

    # Stream across individual manifest rows using progress trackers
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{corpus_name}/{split_name}"):
        audio_path = os.path.join(clips_dir, row['path'])
        try:
            # Extract raw floating point waveform arrays and sample metrics from disk
            audio, sr = sf.read(audio_path)

            # Compress massive raw waveforms into compact OGG byte streams fully in-memory
            ogg_bytes = compress_audio_to_ogg(audio, sr)
            audio_size = len(audio)

            # Standardize transcripts and bundle attributes into our records structure
            batch_records.append({
                'text': normalize_romansh_text(row['sentence']),
                'audio_bytes': ogg_bytes,
                'audio_size': audio_size,
            })
        except Exception as e:
            # Log parsing issues cleanly without halting the broader data orchestration pipelines
            print(f"Error processing {audio_path}: {e}")

        # Once the record buffer limit is reached, flush the contents to a disk file partition
        if len(batch_records) >= BATCH_SIZE:
            write_batch(batch_records, out_dir, part_idx)
            part_idx += 1
            batch_records = []

    # Final sweep check: Flush out any remaining trailing items left over in the buffer
    if batch_records:
        write_batch(batch_records, out_dir, part_idx)

    print("="*60)


print("Converting Romansh data to Omnilingual Parquet (small fragments)")
print("="*60)
print(f"Data root: {DATA_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Batch size: {BATCH_SIZE}, Row group size: {ROW_GROUP_SIZE}")
print("="*60)

# preprocess all idiom folders
for idiom_folder in FOLDER_NAMES:
    corpus_name = get_idiom_name_by_folder(idiom_folder)
    print(f"\nProcessing idiom: {corpus_name} (folder: {idiom_folder})")
    for split in ['train', 'validation', 'test']:
        print(f"  Split: {split}")
        process_split(idiom_folder, split)

print("\nConversion complete!")