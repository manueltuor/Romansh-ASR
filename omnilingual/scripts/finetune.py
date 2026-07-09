"""
Omnilingual Romansh Fine‑Tuning Script
Launches the official Omnilingual ASR training recipe using your custom dataset.

This orchestrator script manages CPU thread allocation, path resolution, environment 
variable configuration, and dynamic YAML synchronization before launching Meta's 
fairseq2-backed training recipe in a managed subprocess.
"""

import os

# Thread limiting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import subprocess
import torch
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent           # scripts/
ROOT_DIR = SCRIPTS_DIR.parent                           # omnilingual/
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"           # submodule root (contains workflows/)

# Duplicate the current system environment variables to isolate modifications
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
# Prepend the submodule root to the environment's PYTHONPATH
env["PYTHONPATH"] = str(SUBMODULE_ROOT) + (f":{pythonpath}" if pythonpath else "")

# Inject the repository root into the current runtime sys.path to safely 
# locate internal packages for the immediate imports below.
sys.path.insert(0, str(ROOT_DIR))

from omnilingual_asr.utils import get_best_gpu, set_config_paths
from omnilingual_asr.constants import LANG_DIST_FILE_ROOT, MODELS_ROOT

#configuration
OUTPUT_DIR = MODELS_ROOT / "omnilingual-ctc-rm-1b-v2"              # target directory
CONFIG_FILE = SUBMODULE_ROOT / "workflows/recipes/wav2vec2/asr/configs/romansh-ctc-finetune.yaml"
DATASET_CARD = SUBMODULE_ROOT / "src/omnilingual_asr/cards/datasets/romansh_dataset.yaml"
stats_file = LANG_DIST_FILE_ROOT

# Synchronize paths: dynamically updates system-specific file paths 
# inside the YAML configurations right before starting execution.
set_config_paths(config_path=CONFIG_FILE, dataset_card_path=DATASET_CARD)

# select best gpu
best_gpu = get_best_gpu()
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using GPU {best_gpu}")
else:
    print("No GPU available – falling back to CPU")

# Verify all required asset descriptors and runtime manifests exist on disk
missing = []
if not DATASET_CARD.exists():
    missing.append(str(DATASET_CARD))
if not stats_file.exists():
    missing.append(str(stats_file))
if not CONFIG_FILE.exists():
    missing.append(str(CONFIG_FILE))

if missing:
    print("Missing required files:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

# Build the programmatic execution command string for the training recipe
cmd = [
    "python", "-m", "workflows.recipes.wav2vec2.asr",
    OUTPUT_DIR,
    "--config-file", str(CONFIG_FILE),
]

print("=" * 60)
print("Starting Omnilingual fine‑tuning")
print(f"   Output dir : {OUTPUT_DIR}")
print(f"   Config     : {CONFIG_FILE}")
print(f"   GPU        : {best_gpu if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# Spawn the recipe inside a decoupled background subprocess. Redirect stderr 
# into stdout to maintain a singular consolidated text log stream.
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,                    # Apply the augmented PYTHONPATH
    cwd=str(ROOT_DIR),          # Establish execution directory context
)

# Actively monitor and flush stdout lines to the terminal in real-time. 
# This prevents log buffering so you can monitor loss and training steps as they happen.
for line in process.stdout:
    print(line, end="")

# Block the parent thread execution until the training engine finishes
process.wait()

if process.returncode == 0:
    print(f"\nTraining finished. Model saved in: {OUTPUT_DIR}")
else:
    print(f"\nTraining failed with code {process.returncode}")
    sys.exit(process.returncode)