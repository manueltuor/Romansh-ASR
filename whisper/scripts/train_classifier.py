# Cell 1: Imports & Setup
import sys
import os
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

notebook_dir = Path.cwd()
whisper_dir = notebook_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import load_all_data, extract_decoder_embeddings, train_classifier
from whisper_asr.utils import get_best_gpu

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#DEVICE = torch.device(f"cuda:{get_best_gpu()}" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cell 2: Load model & processor
MODEL_PATH = "../models/whisper-medium-rm-all-it"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
print("Model loaded.")

# Cell 3: Load data (train and test splits)
train_df = load_all_data("train")
test_df = load_all_data("test")

# Cell 4: Extract embeddings for train and test sets
print("Extracting train embeddings...")
train_embeddings = extract_decoder_embeddings(
    model, processor,
    train_df["audio_path"].tolist(),
    train_df["sentence"].tolist(),
    device=DEVICE, batch_size=8
)
print("Extracting test embeddings...")
test_embeddings = extract_decoder_embeddings(
    model, processor,
    test_df["audio_path"].tolist(),
    test_df["sentence"].tolist(),
    device=DEVICE, batch_size=8
)

# Cell 5: Train classifier and evaluate
train_labels = train_df["idiom"].tolist()
test_labels = test_df["idiom"].tolist()

classifier, predictions = train_classifier(
    train_embeddings, train_labels,
    test_embeddings, test_labels
)

# Optionally save the classifier
import joblib
joblib.dump(classifier, "../models/idiom_classifier.pkl")