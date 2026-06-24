"""
Whisper Encoder Embedding Extraction & Idiom Classification Pipeline.

This script loads a fine-tuned Whisper ASR model, uses its audio encoder block 
as a frozen feature extractor to map raw audio files into downsampled dense vectors, 
and trains a downstream classification head (via joblib/scikit-learn hooks) to 
predict regional Romansh idioms/dialects directly from the acoustic embeddings.
"""

# Imports & Setup
import sys
import os
from pathlib import Path
import torch
import joblib
from transformers import WhisperProcessor, WhisperForConditionalGeneration

script_dir = Path(__file__).resolve().parent
whisper_dir = script_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import load_all_data, train_classifier, extract_encoder_embeddings
from whisper_asr.constants import MODELS_ROOT

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Target path where fine-tuned weights, tokenizers, and configs are archived
MODEL_PATH = MODELS_ROOT / "whisper-medium-rm"
# Pull the text/audio processor pipeline along with the complete transformer layout
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
print("Model loaded.")

# Load test and train data
train_df = load_all_data("train")
test_df = load_all_data("test")

# Extract encoder embeddings for train and test sets
print("Extracting train encoder embeddings...")
train_embeddings = extract_encoder_embeddings(
    model, processor,
    train_df["audio_path"].tolist(),
    device=DEVICE, batch_size=8
)
print("Extracting test encoder embeddings...")
test_embeddings = extract_encoder_embeddings(
    model, processor,
    test_df["audio_path"].tolist(),
    device=DEVICE, batch_size=8
)

# Extract idioms as train and test labels
train_labels = train_df["idiom"].tolist()
test_labels = test_df["idiom"].tolist()

classifier, predictions = train_classifier(
    train_embeddings, train_labels,
    test_embeddings, test_labels
)

# Save classifier
joblib.dump(classifier, MODELS_ROOT / "idiom_classifier.pkl")