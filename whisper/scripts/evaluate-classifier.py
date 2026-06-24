"""
Whisper Embedding Classification Evaluator.

This script loads a fine-tuned Whisper architecture to extract dense acoustic 
embeddings from a Romansh test dataset, utilizes a pre-trained traditional ML 
classifier to predict the spoken idiom, and builds absolute/normalized confusion 
matrices to evaluate regional dialect classification performance.
"""

# Imports & Setup
import sys
import os
from pathlib import Path
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

script_dir = Path(__file__).resolve().parent
whisper_dir = script_dir.parent
sys.path.append(str(whisper_dir))

from whisper_asr import load_all_data, extract_encoder_embeddings
from whisper_asr.utils import get_best_gpu
from whisper_asr.constants import MODELS_ROOT

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device(f"cuda:{get_best_gpu()}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize the fine-tuned Whisper model configuration and processing weights
MODEL_PATH = MODELS_ROOT / "whisper-medium-rm"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
print("Model loaded.")

# load test data and labels (idioms)
test_df = load_all_data("test")
test_labels = test_df["idiom"].tolist()

# Load the downstream classifier module from storage
classifier = joblib.load(MODELS_ROOT / "idiom_classifier.pkl")
# Extract encoder embeddings from test set
test_embeddings = extract_encoder_embeddings(
    model, processor,
    test_df["audio_path"].tolist(),
    device=DEVICE, batch_size=8
)
# Run classification inference over the audio features to generate dialect predictions
predictions = classifier.predict(test_embeddings)
# compute confusion matrix
cm = confusion_matrix(test_labels, predictions, labels=classifier.classes_)

# Wrap the matrix arrays inside a labeled Pandas DataFrame for easier querying/printing
cm_df = pd.DataFrame(cm, index=classifier.classes_, columns=classifier.classes_)
cm_df.index.name = "True idiom"
cm_df.columns.name = "Predicted idiom"

# Report standard raw integer correct/incorrect intersections
print("Absolute counts:")
print(cm_df)

# Generate a row-wise normalized version
cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).round(3)
print("\nNormalized (by true idiom):")
print(cm_norm)
