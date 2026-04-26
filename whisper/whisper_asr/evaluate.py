import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from jiwer import wer, cer
from typing import List, Optional
from evaluate import load as load_metric

class AudioDataset(Dataset):
    """Simple dataset that loads audio files and returns Whisper input features."""
    
    def __init__(self, audio_paths: List[str], processor: WhisperProcessor):
        self.audio_paths = audio_paths
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]
        return input_features

def collate_audio_batch(batch: List[torch.Tensor]) -> torch.Tensor:
    """Pad features to the same length within a batch."""
    max_len = max(f.shape[-1] for f in batch)
    padded_batch = []
    for features in batch:
        pad_len = max_len - features.shape[-1]
        if pad_len > 0:
            padding = torch.zeros((features.shape[0], pad_len))
            padded = torch.cat([features, padding], dim=-1)
        else:
            padded = features
        padded_batch.append(padded)
    return torch.stack(padded_batch)

def transcribe_whisper(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_paths: List[str],
    batch_size: int = 8,
    device: str = "cuda",
    max_length: int = 225,
) -> List[str]:
    """
    Transcribe a list of audio files using a Whisper model.
    
    Args:
        model: Loaded WhisperForConditionalGeneration model.
        processor: Corresponding WhisperProcessor.
        audio_paths: List of paths to audio files.
        batch_size: Batch size for inference.
        device: Device to run inference on.
        max_length: Maximum generation length.
    
    Returns:
        List of transcription strings.
    """
    dataset = AudioDataset(audio_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_audio_batch,
        num_workers=0,
    )
    
    model.to(device)
    model.eval()
    
    transcriptions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Transcribing (Whisper)"):
            batch = batch.to(device)
            predicted_ids = model.generate(
                batch,
                max_length=max_length,
                num_beams=1,
                task="transcribe"
            )
            texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.extend(texts)
    
    return transcriptions

def compute_wer_summary(
    df: pd.DataFrame,
    transcription_col: str,
    reference_col: str = "sentence",
    idiom_col: str = "idiom"
) -> pd.DataFrame:
    """
    Compute per‑idiom WER summary statistics.
    
    Args:
        df: DataFrame containing references, transcriptions, and idiom labels.
        transcription_col: Column name for the hypothesis transcriptions.
        reference_col: Column name for the reference transcriptions.
        idiom_col: Column name for the idiom labels.
    
    Returns:
        DataFrame with per‑idiom WER means, standard deviations, and sample counts.
        DataFrame with added wer and cer scores.
    """
    def compute_wer_safe(ref: str, hyp: str) -> Optional[float]:
        if ref and hyp:
            try:
                return wer(ref, hyp)
            except:
                return None
        return None
    
    def compute_cer_safe(ref: str, hyp: str) -> Optional[float]:
        if ref and hyp:
            try:
                return cer(ref, hyp)
            except:
                return None
        return None
    
    df_copy = df.copy()
    df_copy['wer'] = df_copy.apply(
        lambda row: compute_wer_safe(row[reference_col], row[transcription_col]), axis=1
    )
    df_copy['cer'] = df_copy.apply(
        lambda row: compute_cer_safe(row[reference_col], row[transcription_col]), axis=1
    )
    
    idiom_summary = df_copy.groupby(idiom_col).agg(
        samples=(reference_col, 'count'),
        wer_mean=('wer', 'mean'),
        wer_std=('wer', 'std')
    ).reset_index()
    
    overall = pd.DataFrame({
        idiom_col: ['OVERALL'],
        'samples': [len(df_copy)],
        'wer_mean': [df_copy['wer'].mean()],
        'wer_std': [df_copy['wer'].std()]
    })
    
    summary = pd.concat([idiom_summary, overall], ignore_index=True)
    summary = summary.round(4)
    summary['wer_mean'] *= 100
    summary['wer_std'] *= 100
    return summary, df_copy

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) as a float (0.0 to 1.0).
    Raises ValueError if inputs are empty.
    """
    if not reference or not hypothesis:
        raise ValueError("Reference and hypothesis must be non-empty strings.")
    return wer(reference, hypothesis)

def compute_metrics(pred, tokenizer):
    """
    Compute WER from Seq2SeqTrainer prediction output.
    Requires the tokenizer to decode tokens.
    """
    wer_metric = load_metric("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}