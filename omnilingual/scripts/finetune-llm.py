#!/usr/bin/env python3
"""
Omnilingual Romansh LoRA Fine‑Tuning Script (LLM)

This orchestrator script manages GPU selection, thread limiting, environment 
configuration, and path resolution before launching the official fairseq2-based 
training recipe for the Wav2Vec2‑Llama (LLM) model with LoRA adapters.

It dynamically picks the least‑loaded GPU, synchronises dataset and model card 
paths in the YAML config, and spawns the recipe as a subprocess with the `--lora` 
flag to activate custom LoRA fine‑tuning.
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"

# GPU selection (before torch)
sys.path.insert(0, str(ROOT_DIR))
from omnilingual_asr.utils import get_best_gpu, set_config_paths
from omnilingual_asr.constants import LANG_DIST_FILE_ROOT, MODELS_ROOT
best_gpu = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
print(f"Using GPU {best_gpu}")

# Limit CPU parallelism to avoid resource exhaustion on shared systems
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Import torch after environment variables are set
import torch

# Disable cuDNN (workaround for cuDNN compatibility issues on this server)
torch.backends.cudnn.enabled = False

# Duplicate the current environment to pass to the subprocess
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = str(SUBMODULE_ROOT) + (f":{pythonpath}" if pythonpath else "")

# Configuration – adjust these paths if needed
OUTPUT_DIR = MODELS_ROOT / "omnilingual-llm-rm-1b-v2"
CONFIG_FILE = SUBMODULE_ROOT / "workflows/recipes/wav2vec2/asr/configs/romansh-llm-finetune.yaml"
DATASET_CARD = SUBMODULE_ROOT / "src/omnilingual_asr/cards/datasets/romansh_dataset.yaml"

# Dynamically update file paths inside the YAML config and dataset card
set_config_paths(config_path=CONFIG_FILE, dataset_card_path=DATASET_CARD)

# GPU status check after environment is set up
if torch.cuda.is_available():
    print(f"GPU {best_gpu} ready")
else:
    print("No GPU available – falling back to CPU")

# Pre‑flight checks – ensure all required assets exist before launching
dataset_card = SUBMODULE_ROOT / "src/omnilingual_asr/cards/datasets/romansh_dataset.yaml"
stats_file = LANG_DIST_FILE_ROOT

missing = []
for f in [dataset_card, stats_file, CONFIG_FILE]:
    if not f.exists():
        missing.append(str(f))
if missing:
    print("Missing required files:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

# Build the command to launch the training recipe with LoRA flag
cmd = [
    "python", "-m", "workflows.recipes.wav2vec2.asr",
    OUTPUT_DIR,
    "--config-file", str(CONFIG_FILE),
    "--lora",
]

print("=" * 60)
print("Starting Omnilingual fine‑tuning (Wav2Vec2‑Llama)")
print(f"   Output dir : {OUTPUT_DIR}")
print(f"   Config     : {CONFIG_FILE}")
print(f"   GPU        : {best_gpu if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# Spawn the training recipe in a subprocess, merging stderr into stdout for unified logging
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env, cwd=str(ROOT_DIR))
# Stream the training log line‑by‑line to the terminal in real time
for line in process.stdout:
    print(line, end="")
process.wait()

if process.returncode == 0:
    print(f"\nTraining finished. Model saved in: {OUTPUT_DIR}")
else:
    print(f"\nTraining failed with code {process.returncode}")
    sys.exit(process.returncode)