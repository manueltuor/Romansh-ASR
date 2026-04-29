# whisper_asr/classification.py

import torch
import numpy as np
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import librosa
from typing import List, Optional


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Helper to load audio; also available in utils, but kept here for self‑containedness."""
    audio, sr = librosa.load(path, sr=target_sr)
    return audio


def extract_decoder_embeddings(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_paths: List[str],
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """
    Compute a fixed‑size embedding for each utterance using the **decoder**.
    Steps:
        - Encode audio → encoder_hidden_states
        - Tokenize reference text (with language/task tokens)
        - Run decoder with teacher forcing, output_hidden_states=True
        - Mean‑pool the last‑layer hidden states across all tokens
    Returns:
        numpy array of shape (n_samples, d_model)   (d_model = 1024 for medium)
    """
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting decoder embeddings"):
        batch_audio = audio_paths[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        # 1. Encode audio → input_features (mel)
        input_features = processor(
            [load_audio(p) for p in batch_audio],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        # 2. Get encoder hidden states
        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # (B, T_enc, d)

        # 3. Tokenize reference texts with language/task tokens
        labels_batch = processor(
            text=batch_texts,
            language="it",
            task="transcribe",
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_target_positions,
            padding=True            
        ).input_ids.to(device)

        # 4. Run decoder with teacher forcing
        with torch.no_grad():
            decoder_outputs = model.model.decoder(
                input_ids=labels_batch,
                encoder_hidden_states=encoder_hidden_states,
                output_hidden_states=True
            )
            # decoder_outputs.hidden_states is a tuple of (num_layers+1) tensors,
            # each shaped (B, T_dec, d). Take the last layer: hidden_states[-1]
            last_layer = decoder_outputs.hidden_states[-1]  # (B, T_dec, d)

            # Mean‑pool over the token dimension
            utterance_embeddings = last_layer.mean(dim=1)  # (B, d)

        embeddings.append(utterance_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def train_classifier(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str],
    classifier=None,   # no problematic default; we set it inside if None
):
    """
    Train a classifier and return predictions, accuracy, and the trained model.
    """
    if classifier is None:
        # Simple logistic regression without extra parameters
        classifier = LogisticRegression(max_iter=1000)

    classifier.fit(train_embeddings, train_labels)
    preds = classifier.predict(test_embeddings)

    print("Test Accuracy:", accuracy_score(test_labels, preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, preds))

    cm = confusion_matrix(test_labels, preds, labels=classifier.classes_)
    cm_df = pd.DataFrame(cm, index=classifier.classes_, columns=classifier.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return classifier, preds