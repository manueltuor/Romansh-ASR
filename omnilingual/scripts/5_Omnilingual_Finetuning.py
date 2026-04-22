# %% [markdown]
# # Omnilingual ASR: Romansh Fine-Tuning Pipeline
# This notebook parallelizes Romansh data preparation, registers the dataset 
# for Meta's Omnilingual ASR, and executes fine-tuning on the best available GPU.

# %%
import os
import io
import yaml
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import librosa
import soundfile as sf
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# User provided helpers and constants
from helpers import get_idiom_name_by_folder, get_best_gpu
from constants import DATA_ROOT, FOLDER_NAMES

# %% [markdown]
# ## 1. Hardware & Path Setup
# Configure the GPU and output directories.

# %%
best_gpu = get_best_gpu()
DEVICE = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

OUTPUT_DIR_ARTIFACTS = "./omnilingual-ctc-rm-all"
os.makedirs(OUTPUT_DIR_ARTIFACTS, exist_ok=True)

print("="*60)
print(f"Device: {DEVICE}")
print(f"Output Directory: {OUTPUT_DIR_ARTIFACTS}")
print("="*60)

# %% [markdown]
# ## 2. Parallelized Dataset Preparation
# We use `ProcessPoolExecutor` to speed up audio resampling and OGG compression.
# We use `np.frombuffer(..., dtype=np.int8)` to fix the PyArrow overflow error.

# %%
schema = pa.schema([
    ('text', pa.string()),
    ('audio_bytes', pa.list_(pa.int8())),
    ('audio_size', pa.int64()),
    ('corpus', pa.dictionary(pa.int32(), pa.string())),
    ('split', pa.dictionary(pa.int32(), pa.string())),
    ('language', pa.dictionary(pa.int32(), pa.string()))
])

PARQUET_ROOT = Path("dataset_root_dir/version=0")
PARQUET_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}

def process_single_audio(audio_path, text, corpus_name, split_name, lang_code):
    """Worker function for parallel processing."""
    if not os.path.exists(audio_path):
        return None
    try:
        # Load 16kHz Mono
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio_size = len(wav)
        
        # Compress to OGG bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, 16000, format='ogg')
        raw_bytes = buffer.getvalue()
        
        # FIX: Interpret bytes as signed int8 for PyArrow compatibility
        int8_list = np.frombuffer(raw_bytes, dtype=np.int8).tolist()
        
        return {
            'text': text,
            'audio_bytes': int8_list,
            'audio_size': audio_size,
            'corpus': corpus_name,
            'split': split_name,
            'language': lang_code
        }
    except Exception:
        return None

def process_and_save_partition(split_name, dataframe, corpus_name="romansh", lang_code="roh_Latn"):
    records = []
    num_cores = max(1, os.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {
            executor.submit(
                process_single_audio, 
                row['full_path'], 
                str(row['sentence']), 
                corpus_name, split_name, lang_code
            ): idx for idx, row in dataframe.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split_name.capitalize()}"):
            res = future.result()
            if res: records.append(res)
        
    if not records: return 0
        
    table = pa.Table.from_pandas(pd.DataFrame(records), schema=schema)
    partition_dir = PARQUET_ROOT / f"corpus={corpus_name}" / f"split={split_name}" / f"language={lang_code}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition file indexing
    part_idx = 0
    output_file = partition_dir / f"part-{part_idx}.parquet"
    while output_file.exists():
        part_idx += 1
        output_file = partition_dir / f"part-{part_idx}.parquet"
        
    pq.write_table(table, output_file, row_group_size=100)
    return len(records)

# Run Parallel Ingestion
total_processed = {"train": 0, "dev": 0, "test": 0}
for folder in FOLDER_NAMES:
    idiom_path = Path(DATA_ROOT) / folder
    print(f"\n📂 Idiom: {folder}")
    for tsv_s, target_s in SPLIT_MAP.items():
        p = idiom_path / f"{tsv_s}.tsv"
        if not p.exists(): continue
        df = pd.read_csv(p, sep='\t')
        df['full_path'] = df['path'].apply(lambda x: str(idiom_path / "clips" / x))
        total_processed[target_s] += process_and_save_partition(target_s, df)

print("\n✅ Dataset Conversion Complete.")

# %% [markdown]
# ## 3. Registration & Configuration
# Register the dataset asset card and create the recipe YAML.

# %%
# Asset Card
card_path = Path("src/omnilingual_asr/cards/datasets/romansh_dataset.yaml")
card_path.parent.mkdir(parents=True, exist_ok=True)
with open(card_path, "w") as f:
    f.write("name: romansh_dataset\ndataset_family: mixture_parquet_asr_dataset\n"
            "dataset_config:\n  data: dataset_root_dir/version=0\ntokenizer_ref: omniASR_tokenizer_v1")

# Training Recipe
config_path = Path("workflows/recipes/wav2vec2/asr/configs/romansh-ctc.yaml")
config_path.parent.mkdir(parents=True, exist_ok=True)
config = {
    "dataset": "romansh_dataset",
    "asr_task_config": {"max_audio_len": 960000, "max_num_elements": 7680000},
    "optimizer": {"config": {"lr": 1e-05}},
    "trainer": {"grad_accumulation": 4},
    "regime": {"num_steps": 10000}
}
with open(config_path, "w") as f:
    yaml.dump(config, f)

# %% [markdown]
# ## 4. Fine-Tuning Execution
# Run the training recipe and stream logs.

# %%
cmd = ["python", "-m", "workflows.recipes.wav2vec2.asr", OUTPUT_DIR_ARTIFACTS, "--config-file", str(config_path)]

print(f"🚀 Launching Omnilingual ASR Training on GPU {best_gpu}...")
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())

for line in process.stdout:
    print(line, end="")

process.wait()
if process.returncode == 0:
    print(f"\n✅ Training Finished. Model saved in: {OUTPUT_DIR_ARTIFACTS}")
else:
    print(f"\n❌ Training Error. Code: {process.returncode}")