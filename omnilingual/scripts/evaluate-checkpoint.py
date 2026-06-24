"""
Omnilingual Romansh Evaluation Script from own checkpoint. Set `CHECKPOINT_FILE`
constant to your own checkpoint path.

This script instantiates the base 1B Omnilingual ASR architecture and dynamically
overwrites its parameters with raw state dict weights from your own training 
checkpoint. It then processes the Romansh test partition in micro-batches to 
evaluate dialect-specific performance.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path
from omnilingual_asr.data import load_all_data
from omnilingual_asr.evaluate import add_metrics_columns, idiom_summary, print_evaluation_summary, show_examples
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

SCRIPTS_DIR = Path(__file__).resolve().parent           # scripts/
ROOT_DIR = SCRIPTS_DIR.parent                           # omnilingual/
SUBMODULE_ROOT = ROOT_DIR / "omnilingual_asr"           # submodule root (contains workflows/)

from omnilingual_asr.utils import get_best_gpu, normalize_romansh_text
from omnilingual_asr.constants import MODELS_ROOT

# select best gpu
best_gpu = get_best_gpu()
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using GPU {best_gpu}")
else:
    print("No GPU available – falling back to CPU")

# set your own path
CHECKPOINT_FILE = MODELS_ROOT / "omnilingual-ctc-rm-1b-v2/ws_1.a50a2b74/checkpoints/step_5000/model/pp_00/tp_00/sdp_00.pt" 
LANGUAGE_CODE = "roh_Latn_surs1244"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}, Lang supported: {LANGUAGE_CODE in supported_langs}")

# Initialize the inference pipeline using the BASE model card. 
base_model_card = "omniASR_CTC_1B_v2"
print("Loading base model architecture...")
pipeline = ASRInferencePipeline(model_card=base_model_card, device=DEVICE)

# Load your fine-tuned checkpoint state dict
print("Loading fine-tuned weights...")
checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu")

# Unwrap if needed (common keys: "model" or "module.")
state_dict = checkpoint.get("model", checkpoint)

# Remove "module." prefix if present (from DDP training)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Inject the fine-tuned weights into the pipeline's instantiated model
pipeline.model.load_state_dict(state_dict, strict=False)
pipeline.model.eval()  # Ensure model is in inference mode
print("Model weights loaded successfully!")

# load test data
df_test = load_all_data("test")
# normalize test sentences
df_test["sentence"] = df_test["sentence"].apply(normalize_romansh_text)
print(f"Loaded {len(df_test)} samples")

audio_paths = df_test["audio_path"].tolist()
transcriptions = []

for i in tqdm(range(0, len(audio_paths), BATCH_SIZE)):
    batch = audio_paths[i:i+BATCH_SIZE]
    try:
        # Transcribe the batch.
        results = pipeline.transcribe(
            batch, 
            lang=[LANGUAGE_CODE] * len(batch), 
            batch_size=len(batch)
        )
        transcriptions.extend(results)
    except Exception as e:
        print(f"Batch error at index {i}: {e}")
        transcriptions.extend([""] * len(batch))

df_test["omnilingual_transcription"] = transcriptions
df_test["omnilingual_transcription"] = df_test["omnilingual_transcription"].apply(normalize_romansh_text)
# compute wer and cer
df_test = add_metrics_columns(df_test, "sentence", "omnilingual_transcription")
summary = idiom_summary(df_test)
# print summary
print_evaluation_summary(summary)

#show some examples
show_examples(df_test, "omnilingual_transcription")
