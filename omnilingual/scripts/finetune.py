#!/usr/bin/env python3
"""
Omnilingual Romansh Fine‑Tuning Script
Launches the official Omnilingual ASR training recipe using your custom dataset.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
ROOT_DIR = SCRIPT_DIR.parent                          # omnilingual/
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"         # submodule root (contains workflows/)

# We need to add the submodule root to PYTHONPATH so the subprocess can find 'workflows'
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(SUBMODULE_ROOT) + (f":{pythonpath}" if pythonpath else "")

# Also add to our own sys.path for the helper imports (already installed, but we need it for early imports)
sys.path.insert(0, str(ROOT_DIR))

from omnilingual_asr.utils import get_best_gpu

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OUTPUT_DIR = "./models/omnilingual-ctc-rm-1b"              # where checkpoints/logs go
CONFIG_FILE = SUBMODULE_ROOT / "workflows/recipes/wav2vec2/asr/configs/romansh-ctc-finetune.yaml"

# ----------------------------------------------------------------------
# GPU Selection
# ----------------------------------------------------------------------
best_gpu = get_best_gpu()
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using GPU {best_gpu}")
else:
    print("No GPU available – falling back to CPU")

# ----------------------------------------------------------------------
# Pre‑flight checks
# ----------------------------------------------------------------------
dataset_card = SUBMODULE_ROOT / "src/omnilingual_asr/cards/datasets/romansh_dataset.yaml"
stats_file = Path("/local/scratch/matuor/parquet-dataset/rm-dataset/language_distribution_0.tsv")

missing = []
if not dataset_card.exists():
    missing.append(str(dataset_card))
if not stats_file.exists():
    missing.append(str(stats_file))
if not CONFIG_FILE.exists():
    missing.append(str(CONFIG_FILE))

if missing:
    print("❌ Missing required files:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

# ----------------------------------------------------------------------
# Launch Training
# ----------------------------------------------------------------------
cmd = [
    "python", "-m", "workflows.recipes.wav2vec2.asr",
    OUTPUT_DIR,
    "--config-file", str(CONFIG_FILE),
]

print("=" * 60)
print("🚀 Starting Omnilingual fine‑tuning")
print(f"   Output dir : {OUTPUT_DIR}")
print(f"   Config     : {CONFIG_FILE}")
print(f"   GPU        : {best_gpu if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,                    # PYTHONPATH set here
    cwd=str(ROOT_DIR),
)

for line in process.stdout:
    print(line, end="")

process.wait()

if process.returncode == 0:
    print(f"\n✅ Training finished. Model saved in: {OUTPUT_DIR}")
else:
    print(f"\n❌ Training failed with code {process.returncode}")
    sys.exit(process.returncode)