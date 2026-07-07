"""
Zero-Shot / Fine-Tuned Whisper Evaluation Pipeline for Romansh Dialects.

This script evaluates whisper models on the test set. You can use this for both finetuned
or normal whisper models. For a finetuned model set `MODEL_PATH` as the path to your finetuned
model, otherwise set it to a whisper model e.g. "openai/whisper-medium"
"""

# Imports & Setup
import sys, os
from pathlib import Path
import torch

script_dir = Path(__file__).resolve().parent
whisper_dir = script_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import transcribe_whisper, compute_idiom_results, print_evaluation_results, load_all_data, apply_causal_attention_mask
from whisper_asr.utils import get_best_gpu, normalize_romansh_text
from whisper_asr.constants import MODELS_ROOT
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#MODEL_PATH = "openai/whisper-medium"             # Switch to the base HuggingFace Hub repository URL
MODEL_PATH = MODELS_ROOT / "whisper-medium-rm"    # Path hook for local fine-tuned checkpoints
DEVICE = torch.device(f"cuda:{get_best_gpu()}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
STREAMING = False

print(f"Using device: {DEVICE}")

# Load processing weights and the core Transformer architecture
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

# attention masking
if STREAMING:
    apply_causal_attention_mask(model)

print("Model loaded.")

# Load test data
test_df = load_all_data("test")
audio_paths = test_df["audio_path"]
references = test_df["sentence"]
idioms = test_df["idiom"]

# Transcribe audio
test_df['transcriptions'] = transcribe_whisper(model, processor, audio_paths, batch_size=BATCH_SIZE, device=DEVICE)

# Normalise transcriptions
test_df['transcriptions'] = test_df['transcriptions'].apply(normalize_romansh_text)
transcriptions = test_df['transcriptions']
# Compute wer and cer for each isiom
summary_df, overall_wer, overall_cer, valid_pairs = compute_idiom_results(references, transcriptions, idioms)
print_evaluation_results(summary_df, overall_wer, overall_cer, len(audio_paths), len(valid_pairs))

# Show example sentences
sample_indices = random.sample(range(len(valid_pairs)), min(5, len(valid_pairs)))
for i, idx in enumerate(sample_indices):
    print(f"--- Sample {i} ---")
    print(f"Reference: {references[idx]}")
    print(f"Idiom: {idioms[idx]}")
    print(f"Hypothesis: {transcriptions[idx]}")
    sample_wer, sample_cer = wer(references[idx], transcriptions[idx]), cer(references[idx], transcriptions[idx])
    print(f"Sample WER: {sample_wer:.4f}, Sample CER: {sample_cer:.4f}")
    print("-" * 40)
