#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"

# GPU selection (before torch)
sys.path.insert(0, str(ROOT_DIR))
from omnilingual_asr.utils import get_best_gpu
best_gpu = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
print(f"Using GPU {best_gpu}")

# Thread limiting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["TQDM_DISABLE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Now we can safely import torch
import torch

# Environment for subprocess (PYTHONPATH)
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(SUBMODULE_ROOT) + (f":{pythonpath}" if pythonpath else "")

# Configuration
OUTPUT_DIR = "./models/omnilingual-llm-rm-1b"
CONFIG_FILE = SUBMODULE_ROOT / "workflows/recipes/wav2vec2/asr/configs/romansh-llm-finetune.yaml"

# GPU status
if torch.cuda.is_available():
    print(f"GPU {best_gpu} ready")
else:
    print("No GPU available – falling back to CPU")

# Pre‑flight checks
dataset_card = SUBMODULE_ROOT / "src/omnilingual_asr/cards/datasets/romansh_dataset.yaml"
stats_file = Path("/local/scratch/matuor/parquet-dataset/rm-dataset/language_distribution_0.tsv")

missing = []
for f in [dataset_card, stats_file, CONFIG_FILE]:
    if not f.exists():
        missing.append(str(f))
if missing:
    print("❌ Missing required files:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

# Launch training (still using the ASR recipe, but with LLM model name)
cmd = [
    "python", "-m", "workflows.recipes.wav2vec2.asr",
    OUTPUT_DIR,
    "--config-file", str(CONFIG_FILE),
    "--lora",
]

print("=" * 60)
print("🚀 Starting Omnilingual fine‑tuning (Wav2Vec2‑Llama)")
print(f"   Output dir : {OUTPUT_DIR}")
print(f"   Config     : {CONFIG_FILE}")
print(f"   GPU        : {best_gpu if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env, cwd=str(ROOT_DIR))
for line in process.stdout:
    print(line, end="")
process.wait()

if process.returncode == 0:
    print(f"\n✅ Training finished. Model saved in: {OUTPUT_DIR}")
else:
    print(f"\n❌ Training failed with code {process.returncode}")
    sys.exit(process.returncode)