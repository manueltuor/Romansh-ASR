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
from typing import List


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Helper to load audio"""
    audio, sr = librosa.load(path, sr=target_sr)
    return audio

def extract_encoder_embeddings(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_paths: List[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """
    Compute audio‑only embeddings by mean‑pooling the encoder’s last hidden state.
    Returns:
        numpy array of shape (n_samples, d_model)
    """
    # Deactivate dropout blocks and switch layers into frozen validation behavior
    model.eval()
    embeddings = []

    # Process files sequentially inside configurable chunk batches to manage memory
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting encoder embeddings"):
        batch_audio = audio_paths[i:i+batch_size]
        # Audio → mel features
        input_features = processor(
            [load_audio(p) for p in batch_audio],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        # Run encoder only
        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            # last_hidden_state shape: (B, T_enc, d_model)
            last_hidden = encoder_outputs.last_hidden_state
            # mean-pool across time
            utterance_embeddings = last_hidden.mean(dim=1)  # (B, d_model)
        # Shift the tensor back to host RAM and drop PyTorch track contexts to convert cleanly to NumPy
        embeddings.append(utterance_embeddings.cpu().numpy())
    # Stack the list of micro-batch matrices along the zero-axis into one continuous array
    return np.concatenate(embeddings, axis=0)


def train_classifier(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str],
    classifier=None,
):
    """
    Train a classifier and return predictions, accuracy, and the trained model.

    Args:
        train_embeddings (np.ndarray): High-dimensional training vectors.
        train_labels (list[str]): Target training class labels (e.g., idiom names).
        test_embeddings (np.ndarray): Verification vectors used to score performance.
        test_labels (list[str]): True categorical validation tags.
        classifier (Any | None): Optional scikit-learn estimator instance.

    Returns:
        tuple: (fitted_classifier_model, array_of_predictions)
    """
    if classifier is None:
        # Initialize logistic regression classifier
        classifier = LogisticRegression(max_iter=1000)

    # Fit the linear weights of the classification boundary lines using the training pairs
    classifier.fit(train_embeddings, train_labels)
    # Predict discrete class labels over unseen verification spaces
    preds = classifier.predict(test_embeddings)

    # Report global baseline classification performance metrics
    print("Test Accuracy:", accuracy_score(test_labels, preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, preds))

    # Generate a confusion matrix cross-tabulation mapping classifications
    cm = confusion_matrix(test_labels, preds, labels=classifier.classes_)
    cm_df = pd.DataFrame(cm, index=classifier.classes_, columns=classifier.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return classifier, preds